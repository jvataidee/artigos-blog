"""Prepara a pasta do MSTAR / IU T-72 variants para o ArcGIS Pro (arcgis.learn).

Requer o conjunto de imagens do SDMS (Sensor Data Management System, Força Aérea dos EUA) já baixado e as
imagens JPG organizadas em uma pasta por classe (A04, A05, A07, A10, A32, A62, A63, A64) dentro de `images/`.
Faz duas coisas: grava o arquivo `map.txt`, com o caminho de cada imagem, e atualiza as classes dos arquivos
`esri_accumulated_stats.json` e `esri_model_definition.emd` do modelo de classificação de imagens do ArcGIS.

Uso: python preparar_dados.py CAMINHO/DA/PASTA/T72Variants [PASTA_COM_OS_ARQUIVOS_ESRI]
Não foi reexecutado na reorganização do artigo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CLASSES = ["A04", "A05", "A07", "A10", "A32", "A62", "A63", "A64"]


def gravar_mapa(raiz: Path) -> int:
    """map.txt com o caminho relativo de cada imagem JPG, uma por linha."""
    caminhos = sorted(p.relative_to(raiz).as_posix() for p in (raiz / "images").rglob("*.JPG"))
    (raiz / "map.txt").write_text("\n".join(caminhos) + "\n", encoding="utf-8")
    return len(caminhos)


def atualizar_classes(pasta_esri: Path) -> None:
    novas = {"Classes": [{"ClassValue": i + 1, "ClassName": nome} for i, nome in enumerate(CLASSES)]}
    for nome in ("esri_accumulated_stats.json", "esri_model_definition.emd"):
        arq = pasta_esri / nome
        conteudo = json.loads(arq.read_text(encoding="utf-8"))
        conteudo.update(novas)
        arq.write_text(json.dumps(conteudo), encoding="utf-8")


def main() -> None:
    raiz = Path(sys.argv[1])
    pasta_esri = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd()
    print(gravar_mapa(raiz), "imagens mapeadas em", raiz / "map.txt")
    atualizar_classes(pasta_esri)
    print("classes atualizadas em", pasta_esri)


if __name__ == "__main__":
    main()
