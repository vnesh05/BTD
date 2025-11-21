import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import glob
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_PATH = "best_mri_model_v2.pth"
BATCH_SIZE = 32

class MRI(Dataset):
    def __init__(self, mode='test'):
        tumour_path = "data/test/tumor/*.jpg"
        no_tumour_path = "data/test/no_tumor/*.jpg"
        self.data = glob.glob(tumour_path) + glob.glob(no_tumour_path)
        self.labels = [1]*len(glob.glob(tumour_path)) + [0]*len(glob.glob(no_tumour_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.float32).unsqueeze(0)

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

def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeeperCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    test_loader = DataLoader(MRI('test'), batch_size=BATCH_SIZE, shuffle=False)
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            predicted = (outputs > 0.5).float()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    test_model()
