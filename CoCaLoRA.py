import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset, DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image
import open_clip

# CoCa 모델 로드
def load_coca_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='laion2b_s34b_b79k'
    )
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
        self.data = torch.tensor(self.data)
        self.targets = torch.tensor(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img.numpy())
        if self.cifar10.transform:
            img = self.cifar10.transform(img)
        return img, target

class CoCaLoRA(nn.Module):
    def __init__(self, base_model, num_classes, lora_rank):
        super(CoCaLoRA, self).__init__()
        self.base_model = base_model.visual  # CoCa 모델의 비주얼 파트 사용
        self.fc = nn.Linear(self.base_model.output_dim, num_classes)  # 새로운 Fully Connected Layer 추가

        # LoRA 레이어 추가
        self.lora_a = nn.Parameter(torch.randn(self.base_model.output_dim, lora_rank))
        self.lora_b = nn.Parameter(torch.randn(lora_rank, num_classes))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)  # 출력 Flatten
        lora_out = torch.matmul(x, self.lora_a)
        lora_out = torch.matmul(lora_out, self.lora_b)
        x = self.fc(x)
        return x + lora_out

# CoCa 모델과 전처리 함수 로드
base_model, preprocess = load_coca_model()
model = CoCaLoRA(base_model, num_classes=8, lora_rank=6)  # num_classes와 lora_rank 변경

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# CoCa 전처리 함수로 데이터 로딩
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
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 에포크마다 정확도 계산
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
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
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Final Test Accuracy: {100 * correct / total:.2f} %')
