"""Baixa o conjunto de fraude em cartões de crédito (ULB/Worldline) para dados/creditcard.csv.

Origem: transações de cartões europeus em dois dias de setembro de 2013, publicadas pela ULB (Machine Learning
Group) e espelhadas no armazenamento público do TensorFlow. Os dados originais estão no Kaggle, em
"Credit Card Fraud Detection". Confira a licença na página de origem antes de reutilizar.
Uso: python baixar_dados.py
"""

import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DESTINO = Path(__file__).parent / "dados" / "creditcard.csv"


def main() -> None:
    DESTINO.parent.mkdir(exist_ok=True)
    with urllib.request.urlopen(URL, timeout=600) as r, open(DESTINO, "wb") as f:
        f.write(r.read())
    print("gravado", DESTINO, f"{DESTINO.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
