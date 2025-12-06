import json
import pandas as pd

# JSON 読み込み
with open("functions.json", "r") as f:
    data = json.load(f)

# DataFrame に変換
df = pd.DataFrame(data)

df_ffmpeg = df[df["project"] == "FFmpeg"]
df_QEMU = df[df["project"] == "QEMU"]

df_ffmpeg = df_ffmpeg[["target", "function"]]
df_QEMU = df_QEMU[["target", "function"]]

print(df_ffmpeg.shape)
print(df_QEMU.shape)

# CSVとして保存
# df_ffmpeg.to_csv("FFmpeg_functions.csv", index=False)
# df_QEMU.to_csv("QEMU_functions.csv", index=False)
