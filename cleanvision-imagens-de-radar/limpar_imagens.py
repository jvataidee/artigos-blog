"""Verifica a qualidade de recortes de radar com a CleanVision e separa as imagens sinalizadas.

Uso: python limpar_imagens.py --imagens dados/imagens [--descartar]

Sem --descartar, só relata. Com --descartar, move as imagens sinalizadas (e os arquivos auxiliares do ArcGIS
que acompanham cada JPG) para a pasta "descartadas", ao lado de --imagens, em vez de apagá-las.
As imagens originais (ICEYE) não são públicas, então este script mostra o método, e os números do artigo
estão em resultados/resumo.json.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from cleanvision.imagelab import Imagelab

# o ArcGIS grava estes arquivos ao lado de cada JPG exportado
EXTENSOES = ["jpg", "jpg.aux.xml", "jgw", "jpg.ovr"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagens", type=Path, required=True, help="pasta com os JPG exportados do ArcGIS")
    ap.add_argument("--descartar", action="store_true", help="move as imagens sinalizadas para a pasta descartadas")
    args = ap.parse_args()

    imagelab = Imagelab(data_path=str(args.imagens))
    imagelab.find_issues()
    imagelab.report()

    print(imagelab.issue_summary)
    issues = imagelab.issues

    # colunas booleanas is_<teste>_issue: True quando a imagem tem o problema
    colunas = issues.select_dtypes(include=bool).columns
    sinalizadas = issues[issues[colunas].any(axis=1)]
    print(f"{len(sinalizadas)} de {len(issues)} imagens com pelo menos um problema")

    if not args.descartar:
        return
    destino = args.imagens.parent / "descartadas"
    destino.mkdir(exist_ok=True)
    for caminho in sinalizadas.index:
        base = Path(caminho).with_suffix("")
        for ext in EXTENSOES:
            arquivo = Path(f"{base}.{ext}")
            if arquivo.exists():
                shutil.move(str(arquivo), destino / arquivo.name)
    print("imagens sinalizadas movidas para", destino)


if __name__ == "__main__":
    main()
