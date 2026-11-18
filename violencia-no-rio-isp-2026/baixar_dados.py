"""Baixa a base municipal mensal do ISP-RJ para dados/municipio.csv.

Fonte: Instituto de Segurança Pública do Estado do Rio de Janeiro, https://www.ispdados.rj.gov.br/estatistica.html
O arquivo é atualizado pelo instituto, então uma nova execução traz meses mais recentes e pode revisar os últimos
já publicados. Os números do artigo usam os dados até outubro de 2026. Confira os termos de uso na página de origem.
Uso: python baixar_dados.py
"""

import urllib.request
from pathlib import Path

URL = "https://www.ispdados.rj.gov.br/Arquivos/BaseMunicipioMensal.csv"
DESTINO = Path(__file__).parent / "dados" / "municipio.csv"


def main() -> None:
    DESTINO.parent.mkdir(exist_ok=True)
    with urllib.request.urlopen(URL, timeout=300) as r, open(DESTINO, "wb") as f:
        f.write(r.read())
    print("gravado", DESTINO, f"{DESTINO.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
