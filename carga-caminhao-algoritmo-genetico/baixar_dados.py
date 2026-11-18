"""Baixa as tabelas de produtos e de itens de pedido do Brazilian E-Commerce Public Dataset by Olist e grava em dados/.

A fonte oficial é o Kaggle (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce, licença CC BY-NC-SA 4.0),
que exige conta e token. Por praticidade, o script usa uma cópia pública do mesmo arquivo em um repositório do GitHub.
Se preferir a fonte oficial, baixe o dataset no Kaggle e copie olist_products_dataset.csv, olist_order_items_dataset.csv e
product_category_name_translation.csv para dados/.

Uso: python baixar_dados.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

DADOS = Path(__file__).parent / "dados"
BASE = "https://raw.githubusercontent.com/Ganesh7699/Brazilian-E-Commerce-OList/HEAD/"
ARQUIVOS = ["olist_products_dataset.csv", "olist_order_items_dataset.csv", "product_category_name_translation.csv"]


def main() -> None:
    DADOS.mkdir(exist_ok=True)
    for nome in ARQUIVOS:
        req = urllib.request.Request(BASE + nome, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 (URL fixa e pública)
            (DADOS / nome).write_bytes(r.read())
        print("baixado", nome)


if __name__ == "__main__":
    main()
