#!/usr/bin/env python3
"""
Baseline Model Comparison Experiment (Small Scale)
Trains 3 baseline models (ResNet50, ViT-B/16, HybridModel) on a small subset
"""

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from data.custom_dataset import CustomLungDataset
from configs import DATASET_PATHS
from model import HybridModel
import timm

print("=" * 60)
print("Baseline Model Comparison Experiment (Small Scale)")
print("=" * 60)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets - small subset for quick experiment
print("\nLoading datasets...")
ds1 = CustomLungDataset(DATASET_PATHS['dataset1'], 'dataset1', transform=transform)
ds2 = CustomLungDataset(DATASET_PATHS['dataset2'], 'dataset2', transform=transform)
ds3 = CustomLungDataset(DATASET_PATHS['dataset3'], 'dataset3', transform=transform)

print(f"  Dataset 1 size: {len(ds1)}")
print(f"  Dataset 2 size: {len(ds2)}")
print(f"  Dataset 3 size: {len(ds3)}")

# Combine and sample 200 total (stratified by class)
combined = ConcatDataset([ds1, ds2, ds3])
total_size = len(combined)
print(f"  Combined dataset size: {total_size}")

indices = list(range(total_size))
torch.manual_seed(42)
sampled_indices = indices[:200]  # Quick test on first 200
small_dataset = Subset(combined, sampled_indices)
train_size = int(0.8 * len(small_dataset))
val_size = len(small_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(small_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

print(f"  Train subset: {len(train_subset)}, Val subset: {len(val_subset)}")

train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

# Train function
def train_model(model, name, epochs=5):
    print(f"\n--- Training {name} ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs}: loss = {avg_loss:.4f}")

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"  Validation accuracy: {accuracy:.2%}")
    return accuracy

# 1. ResNet50
print("\n[1/3] ResNet50 Baseline")
resnet = timm.create_model('resnet50', pretrained=True, num_classes=3)
resnet_accuracy = train_model(resnet, "ResNet50", epochs=5)

# 2. ViT
print("\n[2/3] ViT-B/16 Baseline")
vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=3)
vit_accuracy = train_model(vit, "ViT-B/16", epochs=5)

# 3. HybridModel
print("\n[3/3] HybridModel (CNN-Transformer)")
hybrid = HybridModel(num_classes=3, model_dim=128, nhead=4, num_layers=2, dropout=0.1)
hybrid_accuracy = train_model(hybrid, "HybridModel", epochs=5)

# Summary
print("\n" + "=" * 60)
print("Baseline Comparison Summary")
print("=" * 60)
print(f"ResNet50:     {resnet_accuracy:.2%}")
print(f"ViT-B/16:     {vit_accuracy:.2%}")
print(f"HybridModel:  {hybrid_accuracy:.2%}")
print("=" * 60)