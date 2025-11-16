from model import EnhancedBrainTumorCNN
import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------- CONFIG ----------------
train_dir = "data/train"
val_dir = "data/val"
test_dir = "data/test"
img_size = 224
batch_size = 32
epochs = 10
lr = 3e-4
weight_decay = 1e-4
patience = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"✅ Using device: {device}")

# ---------------- TRANSFORMS ----------------
train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ---------------- LOAD DATA ----------------
train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

classes = train_ds.classes
print(f"✅ Classes: {classes}")

# ---------------- MODEL ----------------
model = EnhancedBrainTumorCNN(num_classes=len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scaler = torch.cuda.amp.GradScaler()

# ---------------- EVAL FUNCTION ----------------
def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

def evaluate(model, loader, device, criterion):
    model.eval()
    all_probs, all_labels = [], []
    running_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_labels, all_probs)
    return running_loss / len(loader.dataset), metrics

# ---------------- TRAIN LOOP ----------------
out_dir = f"outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(out_dir, exist_ok=True)
best_f1, epochs_no_improve = -1, 0
history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}
best_path = os.path.join(out_dir, "best.pt")

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * x.size(0)
        pbar.set_postfix(loss=loss.item())

    train_loss = running_loss / len(train_loader.dataset)
    val_loss, val_metrics = evaluate(model, val_loader, device, criterion)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_f1"].append(val_metrics["f1"])
    history["val_auc"].append(val_metrics["roc_auc"])

    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, F1={val_metrics['f1']:.4f}, AUC={val_metrics['roc_auc']:.4f}")

    if val_metrics["f1"] > best_f1:
        best_f1 = val_metrics["f1"]
        torch.save({"model_state": model.state_dict(), "classes": classes}, best_path)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("⚠️ Early stopping triggered.")
            break

# ---------------- SAVE HISTORY ----------------
with open(os.path.join(out_dir, "history.json"), "w") as f:
    json.dump(history, f)

# ---------------- TEST ----------------
ckpt = torch.load(best_path, map_location=device)
model.load_state_dict(ckpt["model_state"])
test_loss, test_metrics = evaluate(model, test_loader, device, criterion)
print("\n✅ Training complete!")
print("📊 Test Metrics:", test_metrics)

# ---------------- PLOT ----------------
plt.figure()
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.legend(); plt.title("Loss Curves")
plt.show()
