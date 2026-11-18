"""Executa os trechos locais de um Markdown e confere as saídas textuais publicadas.

O trecho de download é compilado, mas não executado; usa-se o CSV arquivado.
Uso: python verificar_post.py caminho/para/o/artigo.md
"""
import ast
import contextlib
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

HERE = Path(__file__).parent


def normalize(text):
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def main():
    source = Path(sys.argv[1]).read_text(encoding="utf-8")
    fence = chr(96) * 3
    blocks = re.findall(fence + r"(python|text)\n(.*?)" + fence, source, flags=re.S)
    namespace = {"np": np, "pd": pd, "RandomForestRegressor": RandomForestRegressor, "KFold": KFold}
    verified = 0
    for i, (language, code) in enumerate(blocks):
        if language != "python":
            continue
        ast.parse(code)
        if "catalog.search(" in code:
            count = len(pd.read_csv(HERE / "dados" / "pixel_soja_s2.csv"))
            assert normalize(blocks[i + 1][1]) == f"cenas encontradas: {count}"
            continue
        code = code.replace('"pixel_soja_s2.csv"', repr(str(HERE / "dados" / "pixel_soja_s2.csv")))
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            exec(compile(code, f"trecho_{i}", "exec"), namespace)
        if i + 1 < len(blocks) and blocks[i + 1][0] == "text":
            actual, expected = normalize(capture.getvalue()), normalize(blocks[i + 1][1])
            if actual != expected:
                raise AssertionError(f"Saída divergente no trecho {i}:\n{actual}\nPublicado:\n{expected}")
            verified += 1
    print(f"Trechos Python compilados; {verified} saídas locais conferidas com o Markdown.")


if __name__ == "__main__":
    main()
