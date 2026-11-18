"""Análise descritiva do boi gordo: sazonalidade do preço, abate de fêmeas por trimestre e ciclo pecuário.

Não faz previsão. Usa os mesmos dados de boi.py (preço real do Paraná, abates por tipo de rebanho e IPCA).

- Sazonalidade do preço: decomposição STL do log do preço real mensal (período 12), em duas janelas de anos.
- Abate de fêmeas: participação de fêmeas por trimestre do ano, em duas janelas de anos.
- Ciclo: média móvel de 4 trimestres do preço real e da participação de fêmeas, e os pontos de virada.

Uso: python analise.py [--saida resultados]
Grava sazonal_preco.csv, sazonal_femeas.csv, ciclo.csv e analise.json em --saida.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

sys.path.insert(0, str(Path(__file__).parent))
import boi  # noqa: E402

AQUI = Path(__file__).parent


def preco_real_mensal() -> pd.Series:
    d = AQUI / "dados"
    preco = pd.read_csv(d / "preco_boi_pr.csv", parse_dates=["data"]).set_index("data").preco_arroba
    preco = preco[preco.index >= "1995-01-01"]
    ipca = pd.read_csv(d / "ipca_mensal.csv", parse_dates=["data"]).set_index("data").ipca_var
    indice = (1 + ipca / 100).cumprod()
    real = preco / indice.reindex(preco.index) * indice.iloc[-1]
    real.index = pd.DatetimeIndex(real.index, freq="MS")
    return real


def pontos_de_virada(s: pd.Series, janela: int = 6) -> list[tuple[str, str, float]]:
    """Máximos e mínimos locais de uma série suavizada (o ponto é o maior ou o menor da janela de cada lado)."""
    v = s.values
    out = []
    for i in range(janela, len(v) - janela):
        w = v[i - janela:i + janela + 1]
        if v[i] == w.max():
            out.append((str(s.index[i]), "pico", float(v[i])))
        elif v[i] == w.min():
            out.append((str(s.index[i]), "vale", float(v[i])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="resultados")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    saida = AQUI / args.saida
    saida.mkdir(exist_ok=True)

    # --- sazonalidade do preço (STL no log do preço real mensal)
    real = preco_real_mensal()
    stl = STL(np.log(real), period=12, robust=True).fit()
    sazonal = pd.DataFrame({"mes": stl.seasonal.index.month, "ano": stl.seasonal.index.year, "efeito": np.exp(stl.seasonal.values) - 1})
    linhas = []
    for nome, (a, b) in {"1995 a 2010": (1995, 2010), "2011 a 2026": (2011, 2026)}.items():
        g = sazonal[(sazonal.ano >= a) & (sazonal.ano <= b)].groupby("mes").efeito.mean()
        linhas += [{"janela": nome, "mes": int(m), "efeito_pct": float(v * 100)} for m, v in g.items()]
    pd.DataFrame(linhas).to_csv(saida / "sazonal_preco.csv", index=False)
    forca = 1 - float(np.var(stl.resid) / np.var(stl.resid + stl.seasonal))          # força da sazonalidade (Hyndman)

    # --- abate de fêmeas por trimestre do ano
    q = boi.carregar()
    q["ano"] = [int(str(i)[:4]) for i in q.index]
    q["tri"] = [int(str(i)[5]) for i in q.index]
    linhas = []
    for nome, (a, b) in {"1997 a 2011": (1997, 2011), "2012 a 2026": (2012, 2026)}.items():
        g = q[(q.ano >= a) & (q.ano <= b)].groupby("tri").femeas.mean()
        linhas += [{"janela": nome, "trimestre": int(t), "femeas_pct": float(v * 100)} for t, v in g.items()]
    pd.DataFrame(linhas).to_csv(saida / "sazonal_femeas.csv", index=False)

    # --- ciclo: médias móveis de 4 trimestres e pontos de virada
    suav = pd.DataFrame({"preco_real": q.preco_real.rolling(4).mean(), "femeas": q.femeas.rolling(4).mean()}).dropna()
    suav.to_csv(saida / "ciclo.csv")
    virada_p = pontos_de_virada(suav.preco_real, 8)
    virada_f = pontos_de_virada(suav.femeas, 8)
    anual = q.groupby("ano")[["preco_real", "femeas"]].mean()
    corr_anual = {k: float(anual.preco_real.diff().corr(anual.femeas.shift(k))) for k in range(0, 6)}
    (saida / "analise.json").write_text(json.dumps({
        "forca_sazonal_preco": forca, "viradas_preco": virada_p, "viradas_femeas": virada_f, "corr_anual_variacao_preco_femeas_defasada": corr_anual,
        "meses_mais_altos": {j: sazonal[(sazonal.ano >= a) & (sazonal.ano <= b)].groupby("mes").efeito.mean().sort_values(ascending=False).head(3).round(4).to_dict()
                             for j, (a, b) in {"1995 a 2010": (1995, 2010), "2011 a 2026": (2011, 2026)}.items()},
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("força da sazonalidade do preço:", round(forca, 3))
    print("viradas do preço:", [(a[:7], b) for a, b, _ in virada_p])
    print("viradas da participação de fêmeas:", [(a[:7], b) for a, b, _ in virada_f])
    print(pd.read_csv(saida / "sazonal_preco.csv").pivot(index="mes", columns="janela", values="efeito_pct").round(1).to_string())
    print(pd.read_csv(saida / "sazonal_femeas.csv").pivot(index="trimestre", columns="janela", values="femeas_pct").round(1).to_string())
    print("corr anual (variação do preço x fêmeas defasada em anos):", {k: round(v, 2) for k, v in corr_anual.items()})


if __name__ == "__main__":
    main()
