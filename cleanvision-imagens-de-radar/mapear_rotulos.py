"""Gera o map.txt que associa cada imagem JPG ao seu rótulo PNG (formato esperado no treino do ArcGIS).

Uso: python mapear_rotulos.py --base dados
Espera dados/images/<nome>.jpg e dados/labels/<nome>.png, e grava dados/map.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    args = ap.parse_args()

    with open(args.base / "map.txt", "w", encoding="utf-8") as f:
        for imagem in sorted((args.base / "images").glob("*.jpg")):
            f.write(f"images/{imagem.name} labels/{imagem.stem}.png\n")
    print("map.txt gravado em", args.base)


if __name__ == "__main__":
    main()
