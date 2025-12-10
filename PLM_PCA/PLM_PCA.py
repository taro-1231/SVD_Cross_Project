from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


MODEL = "microsoft/codebert-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device)
model.eval()

# -----------------------------
# 1. データ読み込み
# -----------------------------
df = pd.read_csv("../dataset_Devign/QEMU_functions_anon.csv")  # code + label
texts = df["func"].tolist()
labels = df["target"].tolist()

texts_train = texts
labels_train = labels

df = pd.read_csv("../dataset_Devign/FFmpeg_functions_anon.csv")  # code + label
texts = df["func"].tolist()
labels = df["target"].tolist()

texts_test = texts
labels_test = labels
# -----------------------------
# 2. ラベルの比率を保って 7:3 に分割
# -----------------------------
# texts_train, texts_test, labels_train, labels_test = train_test_split(
#     texts,
#     labels,
#     test_size=0.3,
#     random_state=42,
#     stratify=labels  # ★ラベル比率を保つ
# )

print("Train size:", len(texts_train))
print("Test size :", len(texts_test))

# -----------------------------
# 3. PLM で埋め込み抽出（訓練データ）
# -----------------------------
def encode_texts(text_list):
    emb_list = []
    for text in text_list:
        enc = tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            h = outputs.last_hidden_state[:, 0, :]  # CLS

        emb_list.append(h.cpu().numpy())

    return np.concatenate(emb_list, axis=0)  # (N,768)


print("Encoding training data...")
emb_train = encode_texts(texts_train)
print("Encoding test data...")
emb_test = encode_texts(texts_test)

print("Train emb:", emb_train.shape)
print("Test emb :", emb_test.shape)

# -----------------------------
# 4. PCA（★訓練データで fit、テストは transform のみ）
# -----------------------------
# k = 70
k = 128
pca = PCA(n_components=k)
pca.fit(emb_train)

z_train = pca.transform(emb_train)
z_test = pca.transform(emb_test)

print("Train PCA:", z_train.shape)
print("Test PCA :", z_test.shape)

# -----------------------------
# 5. SVM を学習（訓練データ）
# -----------------------------
svm = SVC(kernel="linear")
svm.fit(z_train, labels_train)

# -----------------------------
# 6. テストデータで評価
# -----------------------------
preds = svm.predict(z_test)
acc = accuracy_score(labels_test, preds)
f1 = f1_score(labels_test, preds)

print("Test Accuracy:", acc)
print("Test F1:", f1)


import matplotlib.pyplot as plt

# 決定関数（分類境界からの距離）
scores = svm.decision_function(z_test)  # shape: (num_test,)

plt.hist(scores, bins=30, alpha=0.7, color="blue")
plt.title("Decision Function Histogram (SVM)")
plt.xlabel("Score (distance from hyperplane)")
plt.ylabel("Count")
plt.grid(True)
plt.savefig(f"hist/histogram_QF_128.png")

plt.show()

