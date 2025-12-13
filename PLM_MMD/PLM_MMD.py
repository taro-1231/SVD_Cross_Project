import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

dim = 64 #128
MODEL_NAME = "microsoft/codebert-base"

# ===================== Dataset =====================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class VulDataset(Dataset):
    def __init__(self, csv_path, max_len=256):
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # ★ 自分のCSVのカラム名に合わせてここを修正 ★
        code = str(self.df.iloc[idx]["func"])
        label = int(self.df.iloc[idx]["target"])

        tokens = tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
        }

# ===================== MMD 関連 =====================

def gaussian_kernel(x, y, sigma_list=[1, 2, 4, 8, 16]):
    x = x.unsqueeze(1)  # (n, 1, d)
    y = y.unsqueeze(0)  # (1, m, d)
    diff = x - y        # (n, m, d)
    dist_sq = (diff ** 2).sum(-1)  # (n, m)

    kernels = 0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma * sigma)
        kernels += torch.exp(-gamma * dist_sq)
    return kernels  # (n, m)


def mmd_loss(x, y, sigma_list=[1, 2, 4, 8, 16]):
    K_xx = gaussian_kernel(x, x, sigma_list)
    K_yy = gaussian_kernel(y, y, sigma_list)
    K_xy = gaussian_kernel(x, y, sigma_list)

    m = x.size(0)
    n = y.size(0)

    if m > 1:
        K_xx = (K_xx.sum() - K_xx.diag().sum()) / (m * (m - 1))
    else:
        K_xx = 0.0
    if n > 1:
        K_yy = (K_yy.sum() - K_yy.diag().sum()) / (n * (n - 1))
    else:
        K_yy = 0.0

    K_xy = K_xy.mean()

    mmd2 = K_xx + K_yy - 2 * K_xy
    return mmd2

# ===================== モデル本体 =====================

class CodeBERTBinaryClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, freeze_encoder=False, proj_dim=dim):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        hidden_size = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden_size, proj_dim)
        self.classifier = nn.Linear(proj_dim, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)

        z = self.proj(cls_emb)
        z = torch.relu(z)

        logits = self.classifier(z).squeeze(-1)      # (batch,)
        return logits, z

# ====================評価========================
def eval_on_target(model, dataloader):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, _ = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().cpu()
            labels = batch["labels"].long().cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, f1


# ===================== ヒストグラム =====================

def plot_confidence_histogram(model, epoch, dim, dataloader, title="Confidence Histogram"):
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            logits, _ = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())

    os.makedirs("hist_dim_MMD0", exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.hist(all_probs, bins=30, range=(0, 1), edgecolor='black')
    plt.title(title)
    plt.xlabel("Prediction confidence (sigmoid output)")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)

    plt.savefig(f"hist_dim_MMD0/{dim}_{epoch+1}_histogram.png")
    plt.show()

    return np.array(all_probs)

# ===================== 学習準備 =====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

source_ds = VulDataset("../dataset_Devign/FFmpeg_functions_anon.csv")
target_ds = VulDataset("../dataset_Devign/QEMU_functions_anon.csv")

source_loader = DataLoader(source_ds, batch_size=16, shuffle=True)
target_loader = DataLoader(target_ds, batch_size=16, shuffle=True)

model = CodeBERTBinaryClassifier(
    freeze_encoder=False,
    proj_dim=dim,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
cls_loss_fn = nn.BCEWithLogitsLoss()

lambda_mmd = 0.0 #0.1
num_epochs = 7

# ===================== 学習ループ =====================

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    target_iter = iter(target_loader)

    for batch_src in source_loader:
        try:
            batch_tgt = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            batch_tgt = next(target_iter)

        # ----- Source -----
        batch_src = {k: v.to(device) for k, v in batch_src.items()}
        logits_src, feat_src = model(
            input_ids=batch_src["input_ids"],
            attention_mask=batch_src["attention_mask"],
        )
        labels_src = batch_src["labels"]
        cls_loss = cls_loss_fn(logits_src, labels_src)

        # ----- Target (特徴だけ) -----
        batch_tgt = {k: v.to(device) for k, v in batch_tgt.items()}
        _, feat_tgt = model(
            input_ids=batch_tgt["input_ids"],
            attention_mask=batch_tgt["attention_mask"],
        )

        # ===== MMD loss =====
        mmd = mmd_loss(feat_src, feat_tgt)

        loss = cls_loss + lambda_mmd * mmd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_src["input_ids"].size(0)
        train_loss = running_loss / len(source_ds)
    # 評価
    tgt_acc, tgt_f1 = eval_on_target(model, target_loader)
    print(f"[Epoch {epoch+1}] train_loss={train_loss:.4f} "
          f"target_acc={tgt_acc:.4f} target_f1={tgt_f1:.4f}")


    # train_loss = running_loss / len(source_ds)
    # print(f"[Epoch {epoch+1}] train_loss={train_loss:.4f}")
    plot_confidence_histogram(
        model,
        epoch,
        dim=dim,
        dataloader=target_loader,
        title="Confidence Histogram"
    )

# ===================== 評価 =====================

# model.eval()
# all_labels, all_preds = [], []
# with torch.no_grad():
#     for batch in target_loader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         logits, _ = model(
#             input_ids=batch["input_ids"],
#             attention_mask=batch["attention_mask"],
#         )
#         probs = torch.sigmoid(logits)
#         preds = (probs >= 0.5).long().cpu()
#         labels = batch["labels"].long().cpu()
#         all_preds.extend(preds.tolist())
#         all_labels.extend(labels.tolist())

# print("acc, f1 = ", accuracy_score(all_labels, all_preds),
#                     f1_score(all_labels, all_preds))
