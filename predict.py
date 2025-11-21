import torch
import torch.nn as nn
import numpy as np
import cv2
import sys

MODEL_PATH = "best_mri_model_v2.pth"

class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(3, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*12*12, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc_model(self.cnn_model(x))

def predict_image(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeeperCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        pred = (output > 0.5).float().item()

    print("\nImage:", image_path)
    print("Prediction:", "Tumor Detected" if pred == 1 else "No Tumor Detected")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_image(sys.argv[1])
