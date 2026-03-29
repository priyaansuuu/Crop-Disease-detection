import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Paths

VAL_DIR = "../data/Validation"
MODEL_PATH = "../model/fine_tuned_crop_disease_model_v2.pth"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform (same normalization as training)
IMG_SIZE = 256
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset & Loader
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Load model
model = models.efficientnet_b3(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(val_dataset.classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Predictions
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=val_dataset.classes, digits=3))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(20, 18))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=val_dataset.classes, yticklabels=val_dataset.classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
