import math
import random
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoTokenizer
import pandas as pd



# =========================================
# 1. Config
# =========================================

@dataclass
class Config:
    plm_name: str = "microsoft/codebert-base"  # 日本語なら "cl-tohoku/bert-base-japanese" とかに変更
    max_length: int = 128

    bottleneck_dim: int = 768 #128       # 次元削減後の次元 d
    rff_dim: int = 1024             # RFF の出力次元 M
    rff_gamma: float = 1.0          # カーネル幅 1 / (2 * sigma^2) 的なやつ

    domain_hidden_dim: int = 128    # ドメイン判別器の中間次元

    batch_size: int = 32
    lr: float = 2e-5
    num_epochs: int = 3
    lambda_da: float = 0.1          # Domain adversarial の重み
    C_margin: float = 1.0           # Max-margin の C 的な係数
    margin: float = 1.0             # hinge のマージン

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    pos_weight: float = 1.0
    decision_threshold : float = 0.55



def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================
# 2. モデル部品
# =========================================

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class RandomFourierFeatures(nn.Module):
    """
    RFF: z -> sqrt(2/M) * [cos(Wz + b), sin(Wz + b)]
    W: (M, d), b: (M,)
    """
    def __init__(self, input_dim: int, rff_dim: int, gamma: float = 1.0, trainable: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.rff_dim = rff_dim
        self.gamma = gamma

        W = torch.randn(rff_dim, input_dim) * math.sqrt(2 * gamma)
        b = torch.rand(rff_dim) * 2 * math.pi

        if trainable:
            self.W = nn.Parameter(W)
            self.b = nn.Parameter(b)
        else:
            self.register_buffer("W", W)
            self.register_buffer("b", b)

        self.scaling = math.sqrt(2.0 / rff_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        proj = F.linear(x, self.W, self.b)  # (batch, M)
        return self.scaling * torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        # 出力: (batch, 2M)


class DANNMaxMarginRFFBinary(nn.Module):
    """
    PLM → Bottleneck → RFF → Max-Margin（二値）
                        ↘ GRL → Domain Discriminator
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # PLM encoder
        self.encoder = AutoModel.from_pretrained(cfg.plm_name)
        plm_hidden_size = self.encoder.config.hidden_size

        # 次元削減ボトルネック
        self.bottleneck = nn.Linear(plm_hidden_size, cfg.bottleneck_dim)

        # RFF
        self.rff = RandomFourierFeatures(
            input_dim=cfg.bottleneck_dim,
            rff_dim=cfg.rff_dim,
            gamma=cfg.rff_gamma,
            trainable=False  # 必要なら True にしてもOK
        )

        # Max-margin 用線形分類器（二値なので 1 出力）
        self.classifier = nn.Linear(2 * cfg.rff_dim, 1)

        # Domain adversarial
        self.grl = GradientReversalLayer(lambda_=1.0)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(cfg.bottleneck_dim, cfg.domain_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.domain_hidden_dim, 1)  # binary domain: source / target
        )

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            h = outputs.pooler_output
        else:
            h = outputs.last_hidden_state[:, 0]
        return h  # (batch, hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lambda_da: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        # 1. PLM encode
        h = self.encode(input_ids, attention_mask)  # (batch, H)

        # 2. Bottleneck
        z = self.bottleneck(h)  # (batch, d)

        # 3. RFF + classifier
        rff_feat = self.rff(z)           # (batch, 2M)
        logits = self.classifier(rff_feat).squeeze(-1)  # (batch,) SVM の f(x)

        # 4. Domain adversarial
        self.grl.lambda_ = lambda_da
        z_rev = self.grl(z)
        domain_logits = self.domain_discriminator(z_rev).squeeze(-1)  # (batch,)

        return {
            "logits": logits,
            "domain_logits": domain_logits,
            "z": z,
            "h": h,
        }


# =========================================
# 3. Loss 関数
# =========================================

# def max_margin_binary_loss(
#     logits: torch.Tensor,
#     labels: torch.Tensor,
#     C: float = 1.0,
#     margin: float = 1.0,
# ) -> torch.Tensor:
#     """
#     二値SVM型 hinge loss
#     labels: 0 or 1 （脆弱 / 非脆弱）
#       → y ∈ {-1, +1} に変換して hinge を取る：
#         L = C * mean( max(0, margin - y * f(x)) )
#     """
#     # labels: (batch,)
#     y = labels.float() * 2.0 - 1.0  # 0→-1, 1→+1
#     # logits: (batch,) = f(x)
#     losses = F.relu(margin - y * logits)
#     return C * losses.mean()

# クラス重み
def max_margin_binary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
    pos_weight: float = 1.0,  # 追加
) -> torch.Tensor:
    """
    labels: 0 or 1 (>=0 のラベルだけ渡す前提)
    pos_weight: 正例 (label=1) の重み
    """
    labels = labels.float()  # (N,)

    # 0/1 → -1/+1
    y = torch.where(labels > 0, 1.0, -1.0)  # (N,)

    # hinge: max(0, margin - y * f(x))
    losses = torch.clamp(margin - y * logits.view(-1), min=0.0)  # (N,)

    # 重み付け: 正例には pos_weight、それ以外は 1.0
    weights = torch.where(labels > 0, pos_weight, 1.0)
    losses = losses * weights

    return losses.mean()



def domain_adversarial_loss(domain_logits: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
    """
    binary domain classification: source = 1, target = 0
    domain_logits: (batch,)
    domain_labels: (batch,) float or long
    """
    domain_labels = domain_labels.float()
    return F.binary_cross_entropy_with_logits(domain_logits, domain_labels)


# =========================================
# 4. Dataset（ここは自分のデータに合わせて差し替え）
# =========================================

class TextDomainDataset(Dataset):
    """
    例:
      texts: list[str]
      labels: list[int] or None (ターゲット側 unlabeled のため)
              0 = 非脆弱, 1 = 脆弱 みたいな感じで
      domains: list[int] (source=1, target=0)
    """
    def __init__(self, texts, labels, domains, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.domains = domains
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        domain = self.domains[idx]
        label = -1 if self.labels is None else self.labels[idx]

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "domain": torch.tensor(domain, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }
        return item


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    domains = torch.stack([b["domain"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "domains": domains,
        "labels": labels,
    }
# -------------------------
# 4.5
# -------------------------
@torch.no_grad()
def evaluate_binary_classification(
    model: DANNMaxMarginRFFBinary,
    data_loader: DataLoader,
    cfg: Config,
) -> Dict[str, float]:
    """
    labels >= 0 のサンプルだけを使って Acc / F1 を計算する
    （ターゲット側などで label=-1 を無視するため）
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_logits = []
    print(cfg.lambda_da)
    for batch in data_loader:
        input_ids = batch["input_ids"].to(cfg.device)
        attention_mask = batch["attention_mask"].to(cfg.device)
        labels = batch["labels"].to(cfg.device)

        # 推論時は lambda_da は 0 で OK（GRL の backward だけに効くので）
        outputs = model(input_ids, attention_mask, lambda_da=0.0)
        logits = outputs["logits"].view(-1)  # (batch,)
        threshold = cfg.decision_threshold
        preds = (logits > threshold).long()  # 0/1 に変換

        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_logits.append(logits.cpu())

    if len(all_labels) == 0:
        return {"acc": 0.0, "f1": 0.0}

    labels = torch.cat(all_labels)  # (N,)
    preds = torch.cat(all_preds)    # (N,)

    # Accuracy
    acc = (preds == labels).float().mean().item()

    # F1 (positive=1 クラスの F1)
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    logits_all = torch.cat(all_logits)
    # print("threshold =", cfg.decision_threshold)
    # print("logits min/max =", float(logits_all.min()), float(logits_all.max()))
    # print("num_pos_preds =", int(preds.sum()), "/", len(preds))


    return {"acc": acc, "f1": f1}


# =========================================
# 5. 学習ループ
# =========================================

def train(cfg: Config):
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.plm_name)

    # ---------------------------------------------
    # ここは自前のデータで置き換えてください
    # source: labeled (脆弱/非脆弱), target: unlabeled
    # ---------------------------------------------
    df_src = pd.read_csv("../source_peg.csv")
    df_tgt = pd.read_csv("../target_png.csv")

    num_pos = (df_src["label"] == 1).sum()
    num_neg = (df_src["label"] == 0).sum()
    pos_weight = num_neg / max(num_pos, 1)
    cfg.pos_weight = pos_weight


    src_dataset = TextDomainDataset(df_src["code"], df_src["label"], [1]*len(df_src), tokenizer, cfg.max_length)
    tgt_dataset = TextDomainDataset(df_tgt["code"], df_tgt["label"], [0]*len(df_tgt), tokenizer, cfg.max_length)

    source_loader = DataLoader(src_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    target_loader = DataLoader(tgt_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)



    model = DANNMaxMarginRFFBinary(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    global_step = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        target_iter = iter(target_loader)

        for batch in source_loader:
            global_step += 1

            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            # ===== Source batch =====
            source_input_ids = batch["input_ids"].to(cfg.device)
            source_attention_mask = batch["attention_mask"].to(cfg.device)
            source_labels = batch["labels"].to(cfg.device)    # 0/1 or -1
            source_domains = batch["domains"].to(cfg.device)  # all 1

            # ラベルが -1 のものは除外（念のため）
            valid_mask = source_labels >= 0
            source_input_ids = source_input_ids[valid_mask]
            source_attention_mask = source_attention_mask[valid_mask]
            source_labels = source_labels[valid_mask]
            source_domains = source_domains[valid_mask]

            # ===== Target batch =====
            target_input_ids = target_batch["input_ids"].to(cfg.device)
            target_attention_mask = target_batch["attention_mask"].to(cfg.device)
            target_domains = target_batch["domains"].to(cfg.device)  # all 0

            # ===== λ_da スケジューリング（DANN定番） =====
            p = global_step / (cfg.num_epochs * len(source_loader))
            lambda_da = cfg.lambda_da * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)

            # ===== Forward: source =====
            out_src = model(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                lambda_da=lambda_da,
            )
            logits_src = out_src["logits"]               # (batch,)
            domain_logits_src = out_src["domain_logits"] # (batch,)

            # Max-margin loss (source only)
            # cls_loss = max_margin_binary_loss(
            #     logits=logits_src,
            #     labels=source_labels,
            #     C=cfg.C_margin,
            #     margin=cfg.margin,
            # )
            cls_loss = max_margin_binary_loss(
                logits=logits_src,
                labels=source_labels,
                margin=1.0,
                pos_weight=cfg.pos_weight,  # ←追加
            )


            # Domain loss (source)
            da_loss_src = domain_adversarial_loss(
                domain_logits_src,
                source_domains,
            )

            # ===== Forward: target (domain only) =====
            out_tgt = model(
                input_ids=target_input_ids,
                attention_mask=target_attention_mask,
                lambda_da=lambda_da,
            )
            domain_logits_tgt = out_tgt["domain_logits"]

            da_loss_tgt = domain_adversarial_loss(
                domain_logits_tgt,
                target_domains,
            )

            da_loss = 0.5 * (da_loss_src + da_loss_tgt)

            # ===== Total loss =====
            loss = cls_loss + cfg.lambda_da * da_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if global_step % 10 == 0:
            #     print(
            #         f"Epoch {epoch+1}/{cfg.num_epochs} "
            #         f"Step {global_step} "
            #         f"Loss={loss.item():.4f} "
            #         f"Cls={cls_loss.item():.4f} "
            #         f"DA={da_loss.item():.4f} "
            #         f"lambda_da={lambda_da:.3f}"
            #     )
         # ===== Epoch 終了時に評価（source 側） =====

        metrics_source = evaluate_binary_classification(
            model, source_loader, cfg
        )

        print(
            f"[Epoch {epoch+1}/{cfg.num_epochs}] "
            f"Source Acc={metrics_source['acc']:.4f} "
            f"F1={metrics_source['f1']:.4f}"
        )
        metrics_tgt = evaluate_binary_classification(model, target_loader, cfg)
        print(f"Target Acc={metrics_tgt['acc']:.4f} F1={metrics_tgt['f1']:.4f}")



    print("Training finished.")
    return model, tokenizer


# =========================================
# 6. エントリポイント
# =========================================

if __name__ == "__main__":
    cfg = Config()
    for th in [0.5]:
        cfg.decision_threshold = th
        for ld in [0.01, 0.05]:
            cfg.lambda_da = ld
            # print(f'lambda_da = {ld}')

        # print(cfg)
            model, tokenizer = train(cfg)
