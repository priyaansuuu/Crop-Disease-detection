# predictor_using_link.py
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import requests
from io import BytesIO
import argparse
import os

# ============================================================
# 🔧 Config
# ============================================================
DATA_DIR = "/home/akash/ML_Model/crop_disease_detection_dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
CKPT_PATH = os.path.join(DATA_DIR, "fine_tuned_crop_disease_model.pth")
IMG_SIZE = 256

# ============================================================
# 🧠 Load Model + Classes
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Load class names same way as training
class_names = datasets.ImageFolder(TRAIN_DIR).classes
num_classes = len(class_names)

# Build same architecture as training
model = models.efficientnet_b3(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, num_classes)

# Load weights
state_dict = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# ============================================================
# 🖼️ Image Preprocessing
# ============================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def load_image(image_path_or_url):
    """Load from local path or URL"""
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        print("🌐 Fetching image from URL...")
        response = requests.get(image_path_or_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(image_path_or_url).convert("RGB")
    return img

# ============================================================
# 🔮 Predict Function
# ============================================================
def predict(image_path_or_url, topk=5):
    img = load_image(image_path_or_url)
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    pred_class = class_names[pred.item()]
    print(f"\n✅ Predicted Class: {pred_class}")
    print(f"📊 Confidence: {conf.item() * 100:.2f}%")

    # Show top-K predictions
    topk_vals, topk_idx = torch.topk(probs, k=min(topk, len(class_names)))
    print("\n🔝 Top Predictions:")
    for i, (idx, val) in enumerate(zip(topk_idx[0], topk_vals[0])):
        print(f"{i+1}. {class_names[idx]} — {val.item() * 100:.2f}%")

# ============================================================
# 🚀 CLI Runner
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop Disease Predictor (Local or URL Image)")
    parser.add_argument("--img", required=True, help="Path or URL to the image")
    args = parser.parse_args()
    predict(args.img)
