"""Baixa os dados abertos do estudo do boi gordo e grava em dados/.

Fontes (todas públicas, sem cadastro):
  - IPEADATA, série DERAL12_PRBGO12: preço médio recebido pelo agricultor, boi gordo, R$/arroba, Paraná, mensal
  - IBGE SIDRA, tabela 1092: animais abatidos por tipo de rebanho (bois, vacas, novilhos, novilhas), trimestral
  - Banco Central SGS: série 3698 (dólar, venda, média mensal) e série 433 (IPCA, variação mensal)

Uso: python baixar_dados.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

DADOS = Path(__file__).parent / "dados"


def get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 (URLs fixas e públicas)
        return json.loads(r.read().decode("utf-8"))


def preco_boi() -> pd.DataFrame:
    v = get("http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='DERAL12_PRBGO12')")["value"]
    d = pd.DataFrame(v)
    d["data"] = pd.to_datetime(d["VALDATA"].str[:10])
    return d.rename(columns={"VALVALOR": "preco_arroba"})[["data", "preco_arroba"]]


def abates() -> pd.DataFrame:
    v = get("https://apisidra.ibge.gov.br/values/t/1092/n1/all/v/284/p/all/c12716/115236/c18/all")[1:]
    d = pd.DataFrame(v)[["D3C", "D5N", "V"]]
    d.columns = ["trimestre", "tipo", "cabecas"]
    d["cabecas"] = pd.to_numeric(d["cabecas"], errors="coerce")
    return d.pivot(index="trimestre", columns="tipo", values="cabecas").reset_index()


def bcb(serie: int, nome: str) -> pd.DataFrame:
    partes = []
    for ini, fim in (("01/01/1994", "31/12/2003"), ("01/01/2004", "31/12/2014"), ("01/01/2015", "31/12/2026")):
        partes += get(f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados?formato=json&dataInicial={ini}&dataFinal={fim}")
    d = pd.DataFrame(partes)
    d["data"] = pd.to_datetime(d["data"], format="%d/%m/%Y")
    d[nome] = pd.to_numeric(d["valor"])
    return d[["data", nome]]


def main() -> None:
    DADOS.mkdir(exist_ok=True)
    preco_boi().to_csv(DADOS / "preco_boi_pr.csv", index=False)
    abates().to_csv(DADOS / "abates_trimestre.csv", index=False)
    bcb(3698, "dolar").to_csv(DADOS / "dolar_mensal.csv", index=False)
    bcb(433, "ipca_var").to_csv(DADOS / "ipca_mensal.csv", index=False)
    print("dados gravados em", DADOS)


if __name__ == "__main__":
    main()
