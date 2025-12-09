from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd

# =====================
# 設定
# =====================
MODEL_NAME = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# Dataset 定義
# =====================
class VulDataset(Dataset):
    def __init__(self, csv_path, max_len=256):
        df = pd.read_csv(csv_path)
        # 論文の Devign 系データセットを想定
        self.codes = df["func"].tolist()
        self.labels = df["target"].tolist()  # 0/1
        self.max_len = max_len

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = str(self.codes[idx])
        label = int(self.labels[idx])

        enc = tokenizer(
            code,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float)  # BCE 用
        return item


# =====================
# CodeBERT ベースラインモデル
# （次元削減など一切なし）
# =====================
class CodeBERTBaseline(nn.Module):
    def __init__(self, model_name=MODEL_NAME, freeze_encoder=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden_size = self.encoder.config.hidden_size  # 768
        # 768 → 1 のシンプルな線形層
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS トークン

        logits = self.classifier(cls_emb).squeeze(-1)  # [batch]

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
            return loss, logits
        return logits

# =======ヒストグラム===========
import matplotlib.pyplot as plt
import numpy as np
import torch
import os


def plot_confidence_histogram(model, epoch, dataloader, title="Confidence Histogram"):
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            # DataLoader から来る dict: input_ids, attention_mask, domains, labels
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())

    os.makedirs("hist", exist_ok=True)  # ここを追加

    plt.figure(figsize=(6, 4))
    plt.hist(all_probs, bins=30, range=(0, 1), edgecolor='black')
    plt.title(title)
    plt.xlabel("Prediction confidence (sigmoid output)")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)

    plt.savefig(f"hist/{epoch+1}_histogram.png")

    plt.show()


    return np.array(all_probs)
# =============================================



# =====================
# 学習 & 評価ループ
# =====================
def train_and_eval(
    source_csv,
    target_csv,
    batch_size_train=16,
    batch_size_eval=32,
    lr=2e-5,
    num_epochs=5,
    freeze_encoder=False,
):
    # --- データ読み込み ---
    train_ds = VulDataset(source_csv)
    target_ds = VulDataset(target_csv)

    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True)
    target_loader = DataLoader(target_ds, batch_size=batch_size_eval, shuffle=False)

    # --- モデル定義 ---
    model = CodeBERTBaseline(freeze_encoder=freeze_encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        # ====== 学習フェーズ ======
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch["input_ids"].size(0)

        train_loss = running_loss / len(train_ds)

        # ====== 検証フェーズ（target ドメイン上） ======
        model.eval()
        val_running_loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in target_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                val_running_loss += loss.item() * batch["input_ids"].size(0)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long().cpu()
                labels = batch["labels"].long().cpu()

                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        val_loss = val_running_loss / len(target_ds)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"val_f1={val_f1:.4f}"
        )
        plot_confidence_histogram(model, epoch, target_loader, title="Confidence Histogram")

    return model




# =====================
# 使い方例
# FFmpeg → QEMU (F→Q) の CodeBERT ベースライン
# =====================
if __name__ == "__main__":
    source_csv = "dataset_Devign/FFmpeg_functions_anon.csv"  # F
    target_csv = "dataset_Devign/QEMU_functions_anon.csv"    # Q
    # df = pd.read_csv(source_csv)
    # df = pd.read_csv(target_csv)
    # print(source_csv)
    # print(target_csv)

    _ = train_and_eval(
        source_csv=source_csv,
        target_csv=target_csv,
        batch_size_train=16,
        batch_size_eval=32,
        lr=2e-5,
        num_epochs=5,
        freeze_encoder=False,  # True にすると「CodeBERT 固定＋線形層だけ学習」
    )
    
