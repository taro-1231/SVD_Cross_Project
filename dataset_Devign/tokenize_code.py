import pandas as pd
import re

# C の予約語（必要に応じて追加してOK）
C_KEYWORDS = {
    "auto","break","case","char","const","continue","default","do","double",
    "else","enum","extern","float","for","goto","if","inline","int","long",
    "register","restrict","return","short","signed","sizeof","static","struct",
    "switch","typedef","union","unsigned","void","volatile","while",
    # よく出る標準関数名はそのまま残したければここに入れる
    # "printf","scanf","malloc","free", ...
}

# 識別子/数値をまとめて拾うパターン
TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+\.\d+|\d+")

def anonymize_code(code: str) -> str:
    """1つの関数コード文字列を匿名化する"""
    if not isinstance(code, str):
        return code

    var_map = {}
    func_map = {}
    var_count = 1
    func_count = 1

    def repl(m: re.Match) -> str:
        nonlocal var_count, func_count
        token = m.group()
        start, end = m.span()

        # 数値リテラルは全部 NUM に
        if token[0].isdigit():
            return "NUM"

        # C の予約語はそのまま
        if token in C_KEYWORDS:
            return token

        # ここから識別子（変数名 or 関数名）
        # このトークンの直後の文字を見て、 '(' なら関数っぽいとみなす
        rest = code[end:]
        rest_strip = rest.lstrip()
        is_func = rest_strip.startswith("(")

        if is_func:
            if token not in func_map:
                func_map[token] = f"FUNC{func_count}"
                func_count += 1
            return func_map[token]
        else:
            if token not in var_map:
                var_map[token] = f"VAR{var_count}"
                var_count += 1
            return var_map[token]

    return TOKEN_RE.sub(repl, code)

# ==== ここから CSV の処理 ====

# 読み込み（ファイル名は適宜変更）
df_ffmpeg = pd.read_csv("FFmpeg_functions.csv")
df_QEMU = pd.read_csv("QEMU_functions.csv")

print(df_ffmpeg.head())
print(df_QEMU.head())

# func カラムを匿名化して新しいカラムに入れる
df_ffmpeg["func"] = df_ffmpeg["func"].apply(anonymize_code)
df_QEMU["func"] = df_QEMU["func"].apply(anonymize_code)
print(df_ffmpeg.head())
print(df_QEMU.head())


# 結果を保存
df_ffmpeg.to_csv("FFmpeg_functions_anon.csv", index=False)
df_QEMU.to_csv("QEMU_functions_anon.csv", index=False)
