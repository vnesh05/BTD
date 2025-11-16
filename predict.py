from model import EnhancedBrainTumorCNN
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os


MODEL_PATH = "outputs/20251111_214030/best.pt"
IMG_SIZE = 224


if len(sys.argv) > 1:
    IMG_PATH = sys.argv[1]
else:
    IMG_PATH = "test_images/sample.jpg"  

if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"❌ Image not found: {IMG_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

ckpt = torch.load(MODEL_PATH, map_location=device)
classes = ckpt["classes"]

model = EnhancedBrainTumorCNN(num_classes=len(classes))
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

print(f"✅ Model loaded successfully! Classes: {classes}")


tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

img = Image.open(IMG_PATH).convert("L")
x = tf(img).unsqueeze(0).to(device)


with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    prob_tumor = float(probs[1])
    pred_label = classes[int(probs.argmax())]

print("\n🧠 Prediction Results:")
print(f"   ➤ Predicted Class: {pred_label}")
print(f"   ➤ Tumor Probability: {prob_tumor:.4f}")
print(f"   ➤ No Tumor Probability: {float(probs[0]):.4f}")
