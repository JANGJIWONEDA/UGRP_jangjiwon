import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset, DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image
import timm

# ViT 모델 로드
def load_vit_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    return model, preprocess

class FilteredCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.cifar10 = datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
        self.data = []
        self.targets = []
        for img, target in zip(self.cifar10.data, self.cifar10.targets):
            if target < 8:  # Only keep classes 0-7
                self.data.append(img)
                self.targets.append(target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.cifar10.transform:
            img = self.cifar10.transform(img)
        return img, target

class ViTFineTuning(nn.Module):
    def __init__(self, base_model, num_classes):
        super(ViTFineTuning, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(self.base_model.head.in_features, num_classes)  # 새로운 Fully Connected Layer 추가
        self.base_model.head = self.fc  # 모델의 헤드 부분을 새로운 FC 레이어로 교체

    def forward(self, x):
        x = self.base_model.forward_features(x)
        x = x[:, 0]  # 첫 번째 클래스 토큰만 사용
        x = self.fc(x)
        return x

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ViT 모델과 전처리 함수 로드
base_model, preprocess = load_vit_model()
model = ViTFineTuning(base_model, num_classes=8).to(device)  # num_classes 변경

# 일부 레이어만 학습시키고 나머지를 고정
for name, param in model.named_parameters():
    if 'head' not in name and 'blocks.10' not in name and 'blocks.11' not in name:  # 마지막 두 블록과 헤드 외의 파라미터는 고정
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # 학습률 스케줄러 추가

# 전처리 함수로 데이터 로딩
train_dataset = FilteredCIFAR10(root='./data', train=True, download=True, transform=preprocess)
test_dataset = FilteredCIFAR10(root='./data', train=False, download=True, transform=preprocess)

# 훈련 및 테스트 데이터셋에서 100개의 샘플 선택
train_subset = Subset(train_dataset, range(100))
test_subset = Subset(test_dataset, range(100))

# 데이터 로더
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=2)

# 훈련 루프 (간단히)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    scheduler.step()  # 학습률 스케줄러 스텝

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 에포크마다 정확도 계산
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy after epoch {epoch+1}: {accuracy:.2f} %')

print('Finished Training')

# 최종 정확도
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Final Test Accuracy: {100 * correct / total:.2f} %')
