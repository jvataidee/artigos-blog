"""Previsão de casos semanais de dengue em 10 capitais, 4 semanas à frente (protocolo.md).

Série prevista: log(1 + casos estimados por semana), do InfoDengue. Modelos: último valor, mesma semana do ano
anterior, ARIMA, ETS (AutoHybTS), Prophet, ARIMA com clima defasado (temperatura e umidade médias, defasagens de 4 e 8
semanas) e LightGBM sobre defasagens do caso e do clima. Origens a cada 8 semanas, prevendo 4 semanas cada. A
validação são as origens anteriores às últimas 52 semanas, e o teste são as origens dentro das últimas 52 semanas.
Nenhum modelo é escolhido a partir da validação, então o teste é só um recorte mais recente do mesmo protocolo.

Uso: python dengue.py [--origens 40] [--saida resultados]
Grava erros_origens.csv, resumo_capitais.csv, resumo_modelos.csv e resumo.json em --saida.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from autohybts import AutoHybridExperiment
from lightgbm import LGBMRegressor
from prophet import Prophet

AQUI = Path(__file__).parent
FH = 4
PASSO = 8
SEMANAS_TESTE = 52
SEMANAS_PROVISORIAS = 4          # as últimas semanas ainda são estimativas instáveis
CLIMA = ["tempmed", "umidmed"]
LAGS_CLIMA = {"tempmed": [4, 8], "umidmed": [4, 8]}
NOMES = {"seasonal_naive": "Mesma semana do ano anterior", "arima": "ARIMA", "exponential_smoothing": "ETS"}
NAIVE = "Último valor"
BASE = NAIVE


def carregar() -> dict[str, pd.DataFrame]:
    d = pd.read_csv(AQUI / "dados" / "dengue_semanal.csv", parse_dates=["data_iniSE"])
    out = {}
    for cap, g in d.groupby("capital"):
        g = g.sort_values("data_iniSE").iloc[:-SEMANAS_PROVISORIAS].set_index("data_iniSE")
        g.index = pd.DatetimeIndex(g.index, freq="W-SUN")
        g["y"] = np.log1p(g["casos_est"])
        g[CLIMA] = g[CLIMA].interpolate(limit_direction="both")
        out[cap] = g[["y"] + CLIMA]
    return out


def prophet_prev(y: pd.Series, origem: pd.Timestamp) -> np.ndarray:
    tr = y[y.index <= origem].rename_axis("ds").reset_index()
    tr.columns = ["ds", "y"]
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(tr)
    fut = pd.DataFrame({"ds": pd.date_range(origem + pd.Timedelta(weeks=1), periods=FH, freq="W-SUN")})
    return m.predict(fut).yhat.values


def lgbm_prev(g: pd.DataFrame, origem: pd.Timestamp) -> np.ndarray:
    """Quatro modelos diretos (um por horizonte), treinados só com o que se conhecia na origem."""
    h = g[g.index <= origem]
    feats = pd.DataFrame(index=h.index)
    for k in range(8):
        feats[f"y_{k}"] = h.y.shift(k)
        for c in CLIMA:
            feats[f"{c}_{k}"] = h[c].shift(k)
    feats["semana"] = h.index.isocalendar().week.astype(int).values
    prev = []
    for passo in range(1, FH + 1):
        alvo = h.y.shift(-passo)
        ok = feats.notna().all(axis=1) & alvo.notna()
        m = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
        m.fit(feats[ok], alvo[ok])
        prev.append(float(m.predict(feats.loc[[origem]])[0]))
    return np.array(prev)


def ic_blocos(v: np.ndarray, bloco: int = 6, n: int = 2000, semente: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(semente)
    k = len(v)
    bloco = min(bloco, k)
    m = []
    for _ in range(n):
        ini = rng.integers(0, k - bloco + 1, size=int(np.ceil(k / bloco)))
        m.append(np.concatenate([v[i:i + bloco] for i in ini])[:k].mean())
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origens", type=int, default=40)
    ap.add_argument("--saida", default="resultados")
    ap.add_argument("--capitais", default="")
    ap.add_argument("--so-resumo", action="store_true", help="só resume o erros_origens.csv já existente em --saida")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    for nome in ("cmdstanpy", "prophet"):
        logging.getLogger(nome).setLevel(logging.WARNING)
    saida = AQUI / args.saida
    saida.mkdir(exist_ok=True)

    series = carregar()
    if args.capitais and not args.so_resumo:
        series = {k: v for k, v in series.items() if k in args.capitais.split(",")}
    linhas = []
    for cap, g in ({} if args.so_resumo else series).items():
        y = g.y
        exp0 = AutoHybridExperiment(fh=FH, session_id=42, test_size=0).fit(y)
        exp1 = AutoHybridExperiment(fh=FH, session_id=42, test_size=0, exog_lags=LAGS_CLIMA).fit(y, g[CLIMA])
        cv0 = exp0.cross_validate(models=list(NOMES), folds=args.origens, step=PASSO)
        cv1 = exp1.cross_validate(models=[exp1.create_model("arima", exog=True)], folds=args.origens, step=PASSO)
        prev: dict[str, dict] = {}
        for r in cv0.results:
            prev[NOMES[r.candidate.base.name]] = {t.origin_time: (np.array(t.y_pred), np.array(t.y_true)) for t in r.trials}
        prev["ARIMA com clima"] = {t.origin_time: (np.array(t.y_pred), np.array(t.y_true)) for t in cv1.results[0].trials}
        origens = sorted(prev["ARIMA"])
        prev[NAIVE] = {o: (np.repeat(y.loc[o], FH), prev["ARIMA"][o][1]) for o in origens}
        prev["Prophet"] = {o: (prophet_prev(y, o), prev["ARIMA"][o][1]) for o in origens}
        prev["LightGBM"] = {o: (lgbm_prev(g, o), prev["ARIMA"][o][1]) for o in origens}
        corte_teste = y.index[-1] - pd.Timedelta(weeks=SEMANAS_TESTE)
        for m, dd in prev.items():
            for o in origens:
                p, t = dd[o]
                linhas.append({"capital": cap, "modelo": m, "origem": f"{o:%Y-%m-%d}", "periodo": "teste" if o > corte_teste else "validação",
                               "mae_ln": float(np.abs(p - t).mean())})
        pd.DataFrame(linhas).to_csv(saida / "erros_origens.csv", index=False)
        print(cap, "pronta:", len(origens), "origens de", f"{origens[0]:%Y-%m}", "a", f"{origens[-1]:%Y-%m}", flush=True)

    e = pd.read_csv(saida / "erros_origens.csv") if args.so_resumo else pd.DataFrame(linhas)
    # erro relativo ao último valor, por capital e período (soma dos erros do modelo sobre soma dos erros do último valor)
    rel = []
    for (cap, per, m), g in e.groupby(["capital", "periodo", "modelo"]):
        base = e[(e.capital == cap) & (e.periodo == per) & (e.modelo == BASE)].mae_ln.sum()
        rel.append({"capital": cap, "periodo": per, "modelo": m, "mae_ln": g.mae_ln.mean(), "razao_ultimo_valor": g.mae_ln.sum() / base})
    rel = pd.DataFrame(rel)
    rel.to_csv(saida / "resumo_capitais.csv", index=False)
    res = []
    for (per, m), g in rel.groupby(["periodo", "modelo"]):
        lo, hi = ic_blocos(g.razao_ultimo_valor.values, bloco=1)
        res.append({"periodo": per, "modelo": m, "mediana_razao": float(g.razao_ultimo_valor.median()), "ic_inf": lo, "ic_sup": hi,
                    "capitais_que_vencem": float((g.razao_ultimo_valor < 1).mean()), "mae_ln": float(g.mae_ln.mean())})
    res = pd.DataFrame(res).sort_values(["periodo", "mediana_razao"])
    res.to_csv(saida / "resumo_modelos.csv", index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    (saida / "resumo.json").write_text(json.dumps({
        "capitais": sorted(e.capital.unique()), "origens": args.origens, "passo_semanas": PASSO, "horizonte": FH, "semanas_teste": SEMANAS_TESTE,
        "semanas": int(len(next(iter(series.values())))),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
