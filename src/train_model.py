import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# CONFIG
DATA_DIR = "/home/akash/ML_Model/crop_disease_detection_dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
VAL_DIR   = os.path.join(DATA_DIR, "Validation")

# ❗ CHANGE 1: Automatically detect number of classes (removed hardcoded NUM_CLASSES)
# -------------------------------------------------------------------------------
train_transforms_dummy = transforms.Compose([transforms.ToTensor()])
train_dataset_dummy = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms_dummy)
NUM_CLASSES = len(train_dataset_dummy.classes)   # <-- UPDATED
# -------------------------------------------------------------------------------

BATCH_SIZE  = 8         
IMG_SIZE    = 256 
EPOCHS      = 20
LR          = 1e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA PIPELINE
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)

print(f"✅ Loaded dataset: {len(train_dataset)} training, {len(val_dataset)} validation images.")
print(f"✅ Classes: {len(train_dataset.classes)}")

# MODEL — EfficientNet-B3
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
weights = EfficientNet_B3_Weights.IMAGENET1K_V1
model = efficientnet_b3(weights=weights)

# ❗ CHANGE 2: Use dynamic class count instead of hardcoded (NUM_CLASSES)
# ----------------------------------------------------------------------
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
# ----------------------------------------------------------------------

model = model.to(DEVICE)

# LOSS / OPTIMIZER / SCHEDULER
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ❗ CHANGE 3: Save to a NEW .pth file so old one stays safe
# ----------------------------------------------------------------------
BEST_MODEL_PATH = "best_crop_disease_model_v2.pth"   # <-- RENAMED
# ----------------------------------------------------------------------

best_acc = 0.0
scaler = torch.cuda.amp.GradScaler()

for epoch in range(EPOCHS):
    print(f"\n🔥 Epoch {epoch+1}/{EPOCHS}")
    model.train()
    running_loss, running_corrects = 0.0, 0

    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = outputs.argmax(1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == labels).sum()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc  = running_corrects.double() / len(train_dataset)
    print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

    # -------- Validation --------
    model.eval()
    val_loss, val_corrects = 0.0, 0
    with torch.no_grad(), torch.cuda.amp.autocast():
        for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += (preds == labels).sum()

    val_loss /= len(val_dataset)
    val_acc = val_corrects.double() / len(val_dataset)
    print(f"✅ Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"💾 Best model saved as {BEST_MODEL_PATH}!")

    scheduler.step()

print(f"\n🏁 Training complete. Best Val Acc: {best_acc:.4f}")
