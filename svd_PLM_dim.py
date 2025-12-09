from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

MODEL_NAME = "microsoft/codebert-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class VulDataset(Dataset):
    def __init__(self, csv_path, max_len=256):
        df = pd.read_csv(csv_path)
        # self.codes = df["code"].tolist()
        # self.labels = df["label"].tolist()  # 0/1
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
        item["labels"] = torch.tensor(label, dtype=torch.float)  # BCE用
        return item

class CodeBERTBinaryClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, freeze_encoder=False, proj_dim=128): #proj_dim追加
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        hidden_size = self.encoder.config.hidden_size
        # 次元削減層
        self.proj = nn.Linear(hidden_size, proj_dim)


        # self.classifier = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(proj_dim, 1)
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS]

        # 次元削減＋簡単な非線形
        z = self.proj(cls_emb)           # [batch, proj_dim]
        z = torch.relu(z)

        # logits = self.classifier(cls_emb).squeeze(-1)
        logits = self.classifier(z).squeeze(-1)
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
            return loss, logits
        return logits

# 例: pegで学習，pngで評価
# train_ds = VulDataset("source_peg.csv")
# print(len(train_ds))

# target_ds   = VulDataset("target_png.csv") #バリデーション用

# 別のテスト
train_ds = VulDataset("dataset_Devign/FFmpeg_functions_anon.csv")
target_ds = VulDataset("dataset_Devign/QEMU_functions_anon.csv") 

# target_ds = VulDataset("dataset_Devign/FFmpeg_functions_anon.csv")
# train_ds = VulDataset("dataset_Devign/QEMU_functions_anon.csv") 



train_loader = DataLoader(train_ds, batch_size=16, shuffle=True) #データセットをバッチごとに分けてくれる。forで回すとバッチごとに送ってくれる
target_loader   = DataLoader(target_ds, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = CodeBERTBinaryClassifier(freeze_encoder=False).to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)



# for epoch in range(15):
#     # ====== 学習フェーズ ======
#     model.train()
#     running_loss = 0.0

#     for batch in train_loader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         loss, _ = model(
#             input_ids=batch["input_ids"],
#             attention_mask=batch["attention_mask"],
#             labels=batch["labels"],
#         )

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * batch["input_ids"].size(0)

#     train_loss = running_loss / len(train_ds)

#     # ====== 検証フェーズ ======
#     model.eval()
#     val_running_loss = 0.0
#     all_labels = []
#     all_preds = []

#     with torch.no_grad():
#         for batch in target_loader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             loss, logits = model(
#                 input_ids=batch["input_ids"],
#                 attention_mask=batch["attention_mask"],
#                 labels=batch["labels"],
#             )

#             val_running_loss += loss.item() * batch["input_ids"].size(0)

#             # ロジット → 確率 → 0.5 で2値化
#             probs = torch.sigmoid(logits)
#             preds = (probs >= 0.5).long().cpu()
#             labels = batch["labels"].long().cpu()

#             all_preds.extend(preds.tolist())
#             all_labels.extend(labels.tolist())

#     val_loss = val_running_loss / len(target_ds)
#     val_acc = accuracy_score(all_labels, all_preds)
#     val_f1 = f1_score(all_labels, all_preds)

#     print(
#         f"Epoch {epoch+1}: "
#         f"train_loss={train_loss:.4f}  "
#         f"val_loss={val_loss:.4f}  "
#         f"val_acc={val_acc:.4f}  "
#         f"val_f1={val_f1:.4f}"
#     )


for proj_dim in [768]:
# for proj_dim in [256, 128, 64, 32, 16]:
    print(f"\n=== proj_dim = {proj_dim} ===")
    model = CodeBERTBinaryClassifier(
        freeze_encoder=False,
        proj_dim=proj_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(5):
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

        # ====== 検証フェーズ ======
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

                # ロジット → 確率 → 0.5 で2値化
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long().cpu()
                labels = batch["labels"].long().cpu()

                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        val_loss = val_running_loss / len(target_ds)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)

        print(
            f"[dim={proj_dim}] Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"val_f1={val_f1:.4f}"
        )
