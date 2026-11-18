"""Classifica imagens fora do treino com o modelo salvo e confere a classe real nos metadados.

Cada imagem de teste vem com um arquivo de metadados sem extensão `.JPG`; a classe real está na linha 13,
nos três últimos caracteres. Compara a classe prevista pelo modelo com a classe real.

Uso: python prever.py CAMINHO/DO/MODELO CAMINHO/DA/PASTA/teste
Não foi reexecutado na reorganização do artigo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from arcgis.learn import FeatureClassifier


def classe_real(arquivo_metadados: Path) -> str:
    return pd.read_fwf(arquivo_metadados, encoding="unicode_escape").iloc[12][0][-3:]


def main() -> None:
    modelo, teste = Path(sys.argv[1]), Path(sys.argv[2])
    clf = FeatureClassifier.from_model(str(modelo))
    for imagem in sorted(teste.glob("*.JPG")):
        meta = imagem.with_suffix("")
        real = classe_real(meta) if meta.exists() else "?"
        categoria, _, probabilidades = clf.predict(str(imagem), visualize=False, gradcam=False)
        print(f"{imagem.name}: classe real {real} | prevista {categoria}")


if __name__ == "__main__":
    main()
