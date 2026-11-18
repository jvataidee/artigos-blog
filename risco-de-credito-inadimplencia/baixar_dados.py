"""Baixa o conjunto de pedidos de crédito para dados/acquisition_train.csv.

Origem: arquivo anonimizado de pedidos de crédito (45.000 linhas, 43 colunas) distribuído em curso de ciência
de dados e publicado em um link do Dropbox. Não há página oficial com licença, então confira a origem e as
condições de uso antes de reutilizar os dados. O link pode sair do ar.
Uso: python baixar_dados.py
"""

import urllib.request
from pathlib import Path

URL = "http://dl.dropboxusercontent.com/s/xn2a4kzf0zer0xu/acquisition_train.csv?dl=1"
DESTINO = Path(__file__).parent / "dados" / "acquisition_train.csv"


def main() -> None:
    DESTINO.parent.mkdir(exist_ok=True)
    with urllib.request.urlopen(URL, timeout=300) as r, open(DESTINO, "wb") as f:
        f.write(r.read())
    print("gravado", DESTINO, f"{DESTINO.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
