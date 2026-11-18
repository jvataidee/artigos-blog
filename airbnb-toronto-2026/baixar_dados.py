"""Baixa os anúncios de Toronto do Inside Airbnb (recorte de 16/09/2026) para dados/listings.csv.gz.

Fonte: https://insideairbnb.com/get-the-data/ (projeto independente, dados públicos e agregados).
O projeto publica novos recortes periodicamente, então o endereço abaixo é o de setembro de 2026. Para outro
recorte, troque a data na URL pelas listadas na página. Confira a licença (Creative Commons) na página de origem.
Uso: python baixar_dados.py
"""

import urllib.request
from pathlib import Path

URL = "https://data.insideairbnb.com/canada/on/toronto/2026-09-16/data/listings.csv.gz"
DESTINO = Path(__file__).parent / "dados" / "listings.csv.gz"


def main() -> None:
    DESTINO.parent.mkdir(exist_ok=True)
    with urllib.request.urlopen(URL, timeout=300) as r, open(DESTINO, "wb") as f:
        f.write(r.read())
    print("gravado", DESTINO, f"{DESTINO.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
