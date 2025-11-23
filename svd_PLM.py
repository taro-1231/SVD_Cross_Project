from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

MODEL_NAME = "microsoft/codebert-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
codebert = AutoModel.from_pretrained(MODEL_NAME)

class VulDataset(Dataset):
    def __init__(self, csv_path, max_len=256):
        df = pd.read_csv(csv_path)
        self.codes = df["code"].tolist()
        self.labels = df["label"].tolist()  # 0/1
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
    def __init__(self, model_name=MODEL_NAME, freeze_encoder=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS]
        logits = self.classifier(cls_emb).squeeze(-1)
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
            return loss, logits
        return logits

# 例: libpngで学習，pegで評価
train_ds = VulDataset("libpng_train.csv")
val_ds   = VulDataset("peg_test.csv")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CodeBERTBinaryClassifier(freeze_encoder=False).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):
    model.train()
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
    # 検証（省略: ロジットにsigmoid→0.5で2値化してF1とか）
