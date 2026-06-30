import os, sys, math, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath('.'))
from cobot_safety_model.data_loader import load_all_trials, DANGER
from cobot_safety_model.features import segment_all_trials

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. LARGER WINDOW SIZE
WINDOW_SIZE = 1.0
STEP_SIZE   = 0.5

all_trials = load_all_trials("data/multiphysio_hrc/DASIG", generate_labels=True, verbose=False)
X_list, y_list, groups_list = [], [], []

for trial in all_trials:
    x_t, y_t = segment_all_trials([trial], WINDOW_SIZE, STEP_SIZE, label_strategy="all", verbose=False)
    if len(x_t) == 0: continue
    
    mean = np.mean(x_t, axis=(0, 1), keepdims=True)
    std  = np.std(x_t, axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0
    x_norm = (x_t - mean) / std

    X_list.append(x_norm)
    
    # Exclude WARNING (1) from being DANGER.
    y_binary = (y_t >= DANGER).astype(np.int64)
    y_window = y_binary.max(axis=1)
    y_list.append(y_window)
    groups_list.extend([trial.subject_id] * len(x_norm))

del all_trials
import gc
gc.collect()

X_all = np.concatenate(X_list, axis=0)
y_all = np.concatenate(y_list, axis=0)
groups = np.array(groups_list)

X_all_dl = torch.tensor(X_all.transpose(0, 2, 1), dtype=torch.float32)
y_all_dl = torch.tensor(y_all, dtype=torch.float32)

def add_derivatives(x):
    pad_x = x[:, :, :1]
    vel = torch.diff(x, dim=2)
    vel = torch.cat([pad_x, vel], dim=2)
    pad_v = vel[:, :, :1]
    acc = torch.diff(vel, dim=2)
    acc = torch.cat([pad_v, acc], dim=2)
    return torch.cat([x, vel, acc], dim=1)

N_FEATURES = 195

class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
        )
    def forward(self, x):
        out = self.net(x)
        if out.shape[2] != x.shape[2]: out = out[:, :, :x.shape[2]]
        return F.gelu(out + x)

class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv1d(N_FEATURES, 128, 1), nn.BatchNorm1d(128), nn.GELU())
        self.blocks = nn.Sequential(*[TCNBlock(128, 3, 2**i, 0.2) for i in range(7)])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.blocks(self.proj(x))
        return self.head(self.gap(x).squeeze(-1)).squeeze(-1)

gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(X_all_dl, y_all_dl, groups))

train_ds = TensorDataset(X_all_dl[train_idx], y_all_dl[train_idx])
val_ds   = TensorDataset(X_all_dl[val_idx],   y_all_dl[val_idx])

# Use pos_weight instead of FocalLoss/WeightedRandomSampler
n_neg = (y_all_dl[train_idx] == 0).sum()
n_pos = (y_all_dl[train_idx] == 1).sum()
pos_weight = torch.tensor([n_neg / n_pos], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False)

model = TCN().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

print("Training TCN on 1.0s windows with BCEWithLogitsLoss...")
best_f1 = 0
for epoch in range(15):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        xb = add_derivatives(xb)
        # Simple noise
        xb = xb + 0.03 * torch.randn_like(xb)
        
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = add_derivatives(xb.to(device))
            probs = torch.sigmoid(model(xb)).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(yb.numpy())
            
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    
    best_t = 0.5
    best_s = 0
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (probs > t).astype(int)
        s = f1_score(targets, preds, average='macro')
        if s > best_s: best_s = s; best_t = t
        
    preds = (probs > best_t).astype(int)
    f1 = f1_score(targets, preds, average='macro')
    if f1 > best_f1: best_f1 = f1
    print(f"Epoch {epoch+1} | Thresh: {best_t:.2f} | Acc: {accuracy_score(targets, preds):.4f} | F1: {f1:.4f} | Rec: {recall_score(targets, preds):.4f} | Prec: {precision_score(targets, preds):.4f}")

