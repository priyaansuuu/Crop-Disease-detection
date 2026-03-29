from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from torchvision import models, transforms
import torch
import torch.nn as nn
from PIL import Image
import requests
from io import BytesIO
import json
import os

# -------------------------------
# CONFIG
# -------------------------------
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "fine_tuned_crop_disease_model_v2.pth"
CLASS_FILE = "class_names.json"

# -------------------------------
# LOAD CLASS NAMES
# -------------------------------
if not os.path.exists(CLASS_FILE):
    raise RuntimeError("❌ class_names.json not found. Please include it in the repo.")

with open(CLASS_FILE, "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)

# -------------------------------
# LOAD MODEL
# -------------------------------
print("🧠 Loading model...")
model = models.efficientnet_b3(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("❌ Model weights (.pth) file not found in the repo.")

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
print("✅ Model loaded successfully.")

# -------------------------------
# TRANSFORM PIPELINE
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# FASTAPI APP
# -------------------------------
app = FastAPI(title="Crop Disease Detection API", version="2.0")

@app.get("/")
def home():
    return {"message": "🌿 Crop Disease Detection API is live!"}


def load_image(image_path_or_url: str):
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        response = requests.get(image_path_or_url, timeout=20)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(image_path_or_url).convert("RGB")
    return img


@app.get("/predict")
def predict(image_path: str = Query(..., description="Image URL or local file path")):
    try:
        img = load_image(image_path)
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)

        return JSONResponse(content={
            "predicted_class": class_names[pred.item()],
            "confidence": round(float(conf.item()) * 100, 2)
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)