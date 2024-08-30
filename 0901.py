# !pip install timm
# !pip install datasets
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import timm
from datasets import load_dataset
import random
from sklearn.preprocessing import LabelEncoder  # 추가

# ViT 모델 로드
def load_vit_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    return model, preprocess

class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.label_encoder = LabelEncoder()
        labels = [item['label'] for item in dataset]  # 모든 레이블 수집
        self.label_encoder.fit(labels)  # 레이블을 숫자로 인코딩

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['image']
        label = item['label']
        if self.transform:
            img = self.transform(img)
        label = self.label_encoder.transform([label])[0]  # 문자열 레이블을 숫자로 변환
        return img, torch.tensor(label, dtype=torch.long)  # 정수형 텐서로 변환하여 반환

class ViTLoRA(nn.Module):
    def __init__(self, base_model, num_classes, lora_rank):
        super(ViTLoRA, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(self.base_model.head.in_features, num_classes)

        # LoRA 레이어 추가
        self.lora_a = nn.Parameter(torch.randn(self.base_model.head.in_features, lora_rank))
        self.lora_b = nn.Parameter(torch.randn(lora_rank, num_classes))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        x = self.base_model.forward_features(x)
        x = x[:, 0]  # 첫 번째 클래스 토큰만 사용
        lora_out = torch.matmul(x, self.lora_a)
        lora_out = torch.matmul(lora_out, self.lora_b)
        x = self.fc(x)
        return x + lora_out

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ViT 모델과 전처리 함수 로드
base_model, preprocess = load_vit_model()
model = ViTLoRA(base_model, num_classes=8, lora_rank=6).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 데이터셋 로드
train_dataset = load_dataset("xodhks/EmoSet118K", split="train")
test_dataset = load_dataset("xodhks/Children_Sketch", split="train")

# 전처리 함수로 데이터 로딩
train_dataset = CustomDataset(train_dataset, transform=preprocess)
test_dataset = CustomDataset(test_dataset, transform=preprocess)

# 전체 훈련 데이터셋 사용
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

# 테스트 데이터셋 크기 설정
test_size = 100
test_indices = random.sample(range(len(test_dataset)), test_size)
test_subset = Subset(test_dataset, test_indices)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=2)

# 훈련 루프
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

    scheduler.step()

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
