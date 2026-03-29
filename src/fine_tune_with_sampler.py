# fine_tune_with_sampler_resume.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm
import numpy as np
import traceback

# ---- Config ----

TRAIN_DIR = "../data/Train"
VAL_DIR = "../data/Validation"

CHECKPOINT_IN = "../model/best_crop_disease_model_v2.pth"     # <-- CHANGED (new base checkpoint name)
CHECKPOINT_OUT = "../model/fine_tuned_crop_disease_model_v2.pth"  # <-- CHANGED (saving as new file)

IMG_SIZE = 256
BATCH_SIZE = 8
LR = 1e-5
EPOCHS = 5
START_EPOCH = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use safer dataloader settings to avoid WSL/CUDA hangs
NUM_WORKERS = 2
PIN_MEMORY = False

# ---- Transforms ----
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---- Datasets ----
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

# ---- Weighted Random Sampler ----
targets = [label for _, label in train_dataset.samples]
class_counts = np.bincount(targets)
class_weights = 1.0 / np.maximum(class_counts, 1)
sample_weights = [class_weights[label] for label in targets]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# ---- Model setup ----
print("Loading EfficientNet-B3...")
model = models.efficientnet_b3(weights=None)

# ❗ CHANGE 1: Auto-detect class count (instead of trusting checkpoint)
num_classes = len(train_dataset.classes)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, num_classes)  # <-- UPDATED
model = model.to(DEVICE)

# ❗ CHANGE 2: Load checkpoint with strict=False to ignore mismatched classifier when classes changed
if os.path.exists(CHECKPOINT_OUT):
    print(f"Resuming from fine-tuned checkpoint: {CHECKPOINT_OUT}")
    model.load_state_dict(torch.load(CHECKPOINT_OUT, map_location=DEVICE), strict=False)  # <-- UPDATED
elif os.path.exists(CHECKPOINT_IN):
    print(f"Loading base checkpoint: {CHECKPOINT_IN}")
    model.load_state_dict(torch.load(CHECKPOINT_IN, map_location=DEVICE), strict=False)   # <-- UPDATED
    START_EPOCH = 0
else:
    print("No checkpoint found. Exiting.")
    raise SystemExit(1)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_acc = 0.0

try:
    for epoch in range(START_EPOCH, EPOCHS):
        print(f"\n=== Fine-tune Epoch {epoch+1}/{EPOCHS} ===")
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in tqdm(train_loader, desc="Train", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Val", leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # ❗ CHANGE 3: Save new checkpoint file instead of old
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_OUT)
            print(f"💾 Saved improved checkpoint → {CHECKPOINT_OUT}")

except Exception as e:
    print("Exception during training:")
    traceback.print_exc()
    try:
        safe_path = "interrupted_finetune_checkpoint_v2.pth"  # <-- RENAMED
        torch.save(model.state_dict(), safe_path)
        print(f"Saved interrupted checkpoint to {safe_path}")
    except Exception as e2:
        print("Failed to save interrupted checkpoint:", e2)
    print("Exiting. Restart after fixing the error.")
