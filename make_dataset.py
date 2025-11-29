import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

root = Path("./dataset")  #自分のフォルダに合わせて変える

rows = []

# 脆弱コード（ラベル1）
for path in (root / "Libpng_vul").glob("**/*"):
    if path.is_file():
        code_text = path.read_text(encoding="utf-8", errors="ignore")
        rows.append({
            "code": code_text,
            "label": 1,
        })

# 非脆弱コード（ラベル0）
for path in (root / "Libpng_non").glob("**/*"):
    if path.is_file():
        code_text = path.read_text(encoding="utf-8", errors="ignore")
        rows.append({
            "code": code_text,
            "label": 0,
        })

df = pd.DataFrame(rows)
df.to_csv("target_png.csv", index=False)
print(df.head())
print("保存しました: libpng_train.csv")




# df_1_1 = pd.read_csv("source_peg.csv")  # CSV名は適宜変更
# df_1_2 = pd.read_csv("target_png.csv")
# df = pd.concat([df_1_1, df_1_2])

# # label 0 と label 1 に分割
# df_0 = df[df['label'] == 0]
# df_1 = df[df['label'] == 1]

# # それぞれ8:2にランダム分割
# train_0, test_0 = train_test_split(df_0, test_size=0.2, shuffle=True, random_state=42)
# train_1, test_1 = train_test_split(df_1, test_size=0.2, shuffle=True, random_state=42)

# # train/testを結合して保存
# train = pd.concat([train_0, train_1]).sample(frac=1, random_state=42)
# test  = pd.concat([test_0, test_1]).sample(frac=1, random_state=42)

# train.to_csv("train_peg.csv", index=False)
# test.to_csv("test_peg.csv", index=False)

# print("Done! → train.csv, test.csv 出力しました")
