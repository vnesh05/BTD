import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import glob
import random

EPOCH = 100
BATCH_SIZE = 32
LR = 0.0001
PATIENCE = 15

class MRI(Dataset):
    def __init__(self, mode='train'):
        if mode == 'train':
            tumour_path = "data/train/tumor/*.jpg"
            no_tumour_path = "data/train/no_tumor/*.jpg"
        elif mode == 'val':
            tumour_path = "data/val/tumor/*.jpg"
            no_tumour_path = "data/val/no_tumor/*.jpg"
        else:
            tumour_path = "data/test/tumor/*.jpg"
            no_tumour_path = "data/test/no_tumor/*.jpg"

        self.data = glob.glob(tumour_path) + glob.glob(no_tumour_path)
        self.labels = [1]*len(glob.glob(tumour_path)) + [0]*len(glob.glob(no_tumour_path))

    def __len__(self):
        return len(self.data)

    def augment(self, img):
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        if random.random() > 0.5:
            angle = random.randint(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h))
        return img

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if "train" in img_path:
            img = self.augment(img)
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

def train_system():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(MRI('train'), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MRI('val'), batch_size=BATCH_SIZE, shuffle=False)
    model = DeeperCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCH):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "best_mri_model_v2.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > PATIENCE:
            break

if __name__ == "__main__":
    train_system()