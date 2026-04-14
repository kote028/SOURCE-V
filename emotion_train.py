import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# =========================
# CONFIG
# =========================
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "emotion_data/train"
VAL_DIR = "emotion_data/train"   # you can split later

# =========================
# TRANSFORMS (🔥 IMPORTANT)
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# DATASET
# =========================
train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_data = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=0)

# 🔥 DEBUG CHECK
print("Dataset size:", len(train_data))


print("Classes:", train_data.classes)

# =========================
# MODEL (🔥 RESNET18)
# =========================
model = models.resnet18(pretrained=True)

# Freeze early layers (optional but useful)
for param in model.parameters():
    param.requires_grad = True

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))
model = model.to(DEVICE)

# =========================
# LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("Dataset size:", len(train_data))

# =========================
# TRAINING LOOP
# =========================
best_acc = 0.0

for epoch in range(EPOCHS):
    print(f"Starting Epoch {epoch+1}")
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):

        if i % 20 == 0:
            print(f"Batch {i}")

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()



        
        

    # =========================
    # VALIDATION
    # =========================
    
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {running_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    # =========================
    # SAVE BEST MODEL
    # =========================
    if val_acc > best_acc:
        best_acc = val_acc
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/emotion_model.pth")
        print("✅ Best model saved!")

print("\n🔥 Training Complete")
print(f"Best Accuracy: {best_acc:.4f}")

# =========================
# SAVE MODEL (OUTSIDE LOOP)
# =========================
import os
os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/emotion_model.pth")
print("✅ Model saved!")