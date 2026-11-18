"""Rodada 2 da dengue (emenda 1 do protocolo): um LightGBM global para as 10 capitais, com a variação do log dos casos e o Rt.

Um único modelo é treinado com as 10 capitais juntas, só com o que se conhecia em cada origem, e prevê a variação do
log dos casos entre a origem e cada uma das 4 semanas seguintes. Duas versões, sem e com o Rt do InfoDengue. Usa as
mesmas origens da primeira rodada e acrescenta as linhas ao erros_origens.csv.

Uso: python dengue_global.py [--saida resultados]
Precisa de dados/dengue_semanal.csv (baixar_dados.py) e de resultados/erros_origens.csv (dengue.py).
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

AQUI = Path(__file__).parent
FH = 4
SEMANAS_PROVISORIAS = 4
SEMANAS_TESTE = 52
CAPITAIS = {"Recife": 2611606, "Rio de Janeiro": 3304557, "São Paulo": 3550308, "Belo Horizonte": 3106200, "Fortaleza": 2304400,
            "Salvador": 2927408, "Goiânia": 5208707, "Manaus": 1302603, "Curitiba": 4106902, "Campo Grande": 5002704}


def baixar_rt() -> pd.DataFrame:
    arq = AQUI / "dados" / "dengue_rt.csv"
    if arq.exists():
        return pd.read_csv(arq, parse_dates=["data_iniSE"])
    partes = []
    for nome, g in CAPITAIS.items():
        url = (f"https://info.dengue.mat.br/api/alertcity?geocode={g}&disease=dengue&format=csv&ew_start=1&ew_end=53&ey_start=2010&ey_end=2026")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 (URL fixa e pública)
            d = pd.read_csv(io.BytesIO(r.read()), parse_dates=["data_iniSE"])
        d["capital"] = nome
        partes.append(d[["capital", "data_iniSE", "Rt"]])
    out = pd.concat(partes)
    out.to_csv(arq, index=False)
    return out


def montar() -> pd.DataFrame:
    d = pd.read_csv(AQUI / "dados" / "dengue_semanal.csv", parse_dates=["data_iniSE"])
    rt = baixar_rt()
    d = d.merge(rt, on=["capital", "data_iniSE"], how="left").sort_values(["capital", "data_iniSE"])
    d["y"] = np.log1p(d.casos_est)
    return d


def atributos(g: pd.DataFrame, com_rt: bool) -> pd.DataFrame:
    """Atributos de uma capital, todos conhecidos na própria semana (só passado)."""
    f = pd.DataFrame(index=g.index)
    for k in (1, 2, 3, 4, 6, 8):
        f[f"dy_{k}"] = g.y - g.y.shift(k)                     # variação do log em k semanas
    f["y"] = g.y
    f["media_8"] = g.y.rolling(8).mean() - g.y
    f["max_52"] = g.y.rolling(52, min_periods=26).max() - g.y   # distância ao maior nível do último ano
    sem = g.data_iniSE.dt.isocalendar().week.astype(int)
    f["sen"], f["cos"] = np.sin(2 * np.pi * sem / 52), np.cos(2 * np.pi * sem / 52)
    f["tempmed"], f["umidmed"] = g.tempmed.values, g.umidmed.values
    if com_rt:
        f["rt"] = g.Rt.values
    return f


def prever(d: pd.DataFrame, origem: pd.Timestamp, com_rt: bool) -> dict[str, np.ndarray]:
    """Um LightGBM por horizonte, treinado nas 10 capitais com a informação até a origem."""
    trein, atual = [], {}
    for cap, g in d.groupby("capital"):
        g = g[g.data_iniSE <= origem].reset_index(drop=True)
        f = atributos(g, com_rt)
        trein.append((cap, g, f))
        atual[cap] = f.iloc[[-1]]
    prev = {cap: np.zeros(FH) for cap in atual}
    for passo in range(1, FH + 1):
        Xs, ys = [], []
        for cap, g, f in trein:
            alvo = g.y.shift(-passo) - g.y                      # variação até a semana origem + passo
            ok = f.notna().all(axis=1) & alvo.notna()
            Xs.append(f[ok])
            ys.append(alvo[ok])
        X, y = pd.concat(Xs), pd.concat(ys)
        m = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
        m.fit(X, y)
        for cap in atual:
            ultimo = trein[[c for c, _, _ in trein].index(cap)][1].y.iloc[-1]
            prev[cap][passo - 1] = ultimo + float(m.predict(atual[cap].fillna(0.0))[0])
    return prev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="resultados")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    saida = AQUI / args.saida

    d = montar()
    # as mesmas semanas da primeira rodada: sem as 4 provisórias
    d = pd.concat([g.iloc[:-SEMANAS_PROVISORIAS] for _, g in d.groupby("capital")], ignore_index=True)
    e = pd.read_csv(saida / "erros_origens.csv")
    e = e[~e.modelo.str.startswith("LightGBM global")]
    origens = sorted(pd.to_datetime(e.origem.unique()))
    corte_teste = d.data_iniSE.max() - pd.Timedelta(weeks=SEMANAS_TESTE)
    linhas = []
    for com_rt, nome in ((False, "LightGBM global"), (True, "LightGBM global com Rt")):
        for o in origens:
            prev = prever(d, o, com_rt)
            for cap, p in prev.items():
                g = d[d.capital == cap].set_index("data_iniSE")
                real = g.y.reindex(pd.date_range(o + pd.Timedelta(weeks=1), periods=FH, freq="W-SUN")).values
                linhas.append({"capital": cap, "modelo": nome, "origem": f"{o:%Y-%m-%d}", "periodo": "teste" if o > corte_teste else "validação",
                               "mae_ln": float(np.abs(p - real).mean())})
        print(nome, "pronto", flush=True)
    out = pd.concat([e, pd.DataFrame(linhas)], ignore_index=True)
    out.to_csv(saida / "erros_origens.csv", index=False)
    print("erros_origens.csv atualizado:", len(out), "linhas")


if __name__ == "__main__":
    main()
