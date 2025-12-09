import json
import pandas as pd

# JSON 読み込み
with open("function.json", "r") as f:
    data = json.load(f)

# DataFrame に変換
df = pd.DataFrame(data)
# print(df.shape)

# print(df["project"].unique())

df_ffmpeg = df[df["project"] == "FFmpeg"]
df_QEMU = df[df["project"] == "qemu"]

df_ffmpeg = df_ffmpeg[["target", "func"]]
df_QEMU = df_QEMU[["target", "func"]]



print(df_ffmpeg.shape)
print(df_QEMU.shape)

# CSVとして保存
df_ffmpeg.to_csv("FFmpeg_functions.csv", index=False)
df_QEMU.to_csv("QEMU_functions.csv", index=False)
