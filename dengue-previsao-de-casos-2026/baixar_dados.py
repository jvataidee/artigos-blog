"""Baixa os casos semanais de dengue e o clima de 10 capitais, do InfoDengue (Fiocruz/FGV), e grava em dados/.

Fonte aberta, sem cadastro: https://info.dengue.mat.br (API alertcity). Cada linha é uma semana epidemiológica, com
casos notificados (casos), casos estimados com atraso de notificação corrigido (casos_est) e clima semanal
(temperatura, umidade).

Uso: python baixar_dados.py
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

DADOS = Path(__file__).parent / "dados"
CAPITAIS = {
    "Recife": 2611606, "Rio de Janeiro": 3304557, "São Paulo": 3550308, "Belo Horizonte": 3106200,
    "Fortaleza": 2304400, "Salvador": 2927408, "Goiânia": 5208707, "Manaus": 1302603, "Curitiba": 4106902,
    "Campo Grande": 5002704,
}


def baixar(geocode: int) -> pd.DataFrame:
    url = (f"https://info.dengue.mat.br/api/alertcity?geocode={geocode}&disease=dengue&format=csv"
           "&ew_start=1&ew_end=53&ey_start=2010&ey_end=2026")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 (URL fixa e pública)
        return pd.read_csv(io.BytesIO(r.read()))


def main() -> None:
    DADOS.mkdir(exist_ok=True)
    partes = []
    for nome, g in CAPITAIS.items():
        d = baixar(g)
        d["capital"] = nome
        partes.append(d[["capital", "data_iniSE", "SE", "casos", "casos_est", "pop", "tempmin", "tempmed", "tempmax", "umidmed"]])
        print(nome, len(d), d.data_iniSE.min(), d.data_iniSE.max(), flush=True)
    pd.concat(partes).to_csv(DADOS / "dengue_semanal.csv", index=False)
    print("dados gravados em", DADOS)


if __name__ == "__main__":
    main()
