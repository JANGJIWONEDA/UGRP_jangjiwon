import math

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Subset

class ResNetLoRA(nn.Module):
    def __init__(self, base_model, num_classes, lora_rank):
        super(ResNetLoRA, self).__init__()
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])  # Remove the final fully connected layer
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)  # New fully connected layer

        # Adding LoRA layers
        self.lora_a = nn.Parameter(torch.randn(base_model.fc.in_features, lora_rank))
        self.lora_b = nn.Parameter(torch.randn(lora_rank, num_classes))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))  # Using math.sqrt(5)
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        lora_out = torch.matmul(x, self.lora_a)
        lora_out = torch.matmul(lora_out, self.lora_b)
        x = self.fc(x)
        return x + lora_out

# Model, Loss, Optimizer
base_model = models.resnet18(pretrained=True)
model = ResNetLoRA(base_model, num_classes=10, lora_rank=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data loading
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the full CIFAR-10 dataset
full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
full_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Select 100 samples from the training and test datasets
train_subset = Subset(full_train_dataset, range(100))
test_subset = Subset(full_test_dataset, range(100))

# Data loaders
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=2)

# Training loop (simplified)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f'Test Accuracy: {100 * correct / total} %')
