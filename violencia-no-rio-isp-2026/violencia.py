"""Violência no estado do Rio de Janeiro, 2014 a outubro de 2026: tendência, sazonalidade e previsão de 12 meses.

Dados: BaseMunicipioMensal do ISP-RJ (Instituto de Segurança Pública), ocorrências registradas por município e
mês, 92 municípios. São registros de ocorrências, e não a violência total. Duas séries estaduais: letalidade
violenta (soma de homicídios dolosos, latrocínios, lesões corporais seguidas de morte e mortes por intervenção
policial) e total de roubos.
Previsão (protocolo v2, ver protocolo.md): 12 meses à frente, em 84 origens mensais (2018-12 a 2025-10). Compara a
referência sazonal, ARIMA e ETS (AutoHybTS), ARIMA e Prophet com a pandemia e a intervenção federal de 2018 como
eventos, e a combinação (média) de ETS, ARIMA com eventos e Prophet. A comparação é pareada por origem, com
intervalo de confiança por bootstrap em blocos de 12 origens.

Uso: python violencia.py [--saida resultados]
Grava resumo.json, anual_regiao.csv, sazonalidade.csv, validacao_origens.csv e validacao_resumo.csv em --saida.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from autohybts import AutoHybridExperiment
from prophet import Prophet

AQUI = Path(__file__).parent
FH = 12
SERIES = {"letalidade_violenta": "Letalidade violenta", "total_roubos": "Roubos (total)"}
ORIGENS = 84
EVENTOS = ["pandemia", "intervencao"]
REF = "Mesmo mês do ano anterior"
MODELOS = {"seasonal_naive": REF, "arima": "ARIMA", "exponential_smoothing": "ETS"}


def eventos(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Variáveis de evento (0 ou 1), conhecidas no futuro: pandemia (2020-03 a 2021-12) e intervenção federal (2018-02 a 2018-12)."""
    return pd.DataFrame({
        "pandemia": ((idx >= "2020-03-01") & (idx <= "2021-12-01")).astype(float),
        "intervencao": ((idx >= "2018-02-01") & (idx <= "2018-12-01")).astype(float),
    }, index=idx)


def wape(y_pred, y_true) -> float:
    return float(np.abs(np.asarray(y_true) - np.asarray(y_pred)).sum() / np.asarray(y_true).sum())


def ic_blocos(dif: np.ndarray, bloco: int = 12, n: int = 2000, semente: int = 0) -> tuple[float, float]:
    """Intervalo de 95% da média da diferença, com bootstrap em blocos (origens vizinhas são dependentes)."""
    rng = np.random.default_rng(semente)
    k = len(dif)
    medias = []
    for _ in range(n):
        ini = rng.integers(0, k - bloco + 1, size=int(np.ceil(k / bloco)))
        medias.append(np.concatenate([dif[i:i + bloco] for i in ini])[:k].mean())
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def carregar() -> pd.DataFrame:
    d = pd.read_csv(AQUI / "dados" / "municipio.csv", sep=";", encoding="latin-1")
    d["data"] = pd.to_datetime(dict(year=d.ano, month=d.mes, day=1))
    return d


def serie(d: pd.DataFrame, col: str) -> pd.Series:
    s = d.groupby("data")[col].sum()
    s.index = pd.DatetimeIndex(s.index, freq="MS")
    return s


def prophet_prev(s: pd.Series, origem: pd.Timestamp) -> np.ndarray:
    ev = eventos(s.index)
    tr = pd.concat([s.rename("y"), ev], axis=1)[s.index <= origem].rename_axis("ds").reset_index()
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    for e in EVENTOS:
        m.add_regressor(e)
    m.fit(tr)
    fut_idx = pd.date_range(origem + pd.offsets.MonthBegin(1), periods=FH, freq="MS")
    fut = pd.concat([pd.Series(fut_idx, name="ds"), eventos(fut_idx).reset_index(drop=True)], axis=1)
    return m.predict(fut).yhat.values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="resultados")
    args = ap.parse_args()
    for nome in ("cmdstanpy", "prophet"):
        logging.getLogger(nome).setLevel(logging.WARNING)
    saida = AQUI / args.saida
    saida.mkdir(exist_ok=True)

    d = carregar()
    print(f"{len(d)} linhas | {d.fmun.nunique()} municípios | de {d.data.min():%Y-%m} a {d.data.max():%Y-%m}")

    # exploração: ano a ano por região e sazonalidade
    anual = d.groupby([d.data.dt.year.rename("ano"), "regiao"])[list(SERIES)].sum().reset_index()
    anual.to_csv(saida / "anual_regiao.csv", index=False)
    full = d[d.data.dt.year <= 2025]
    saz = []
    for col in SERIES:
        s = serie(full, col)
        share = s.groupby([s.index.year, s.index.month]).sum().unstack() \
            .div(s.groupby(s.index.year).sum(), axis=0) * 12      # 1,0 é o mês médio do ano
        saz.append(pd.DataFrame({"serie": SERIES[col], "mes": range(1, 13), "indice": share.mean().values}))
    pd.concat(saz).to_csv(saida / "sazonalidade.csv", index=False)

    linhas = []
    for col, nome_s in SERIES.items():
        s = serie(d, col)
        X = eventos(s.index)
        exp0 = AutoHybridExperiment(fh=FH, session_id=42, test_size=0).fit(s)
        exp1 = AutoHybridExperiment(fh=FH, session_id=42, test_size=0, exog_future=tuple(EVENTOS)).fit(s, X)
        cv0 = exp0.cross_validate(models=list(MODELOS), folds=ORIGENS, step=1)
        cv1 = exp1.cross_validate(models=[exp1.create_model("arima", exog=True)], folds=ORIGENS, step=1)
        prev: dict[str, dict] = {}
        for r in cv0.results:
            prev[MODELOS[r.candidate.base.name]] = {t_.origin_time: (np.array(t_.y_pred), np.array(t_.y_true)) for t_ in r.trials}
        prev["ARIMA com eventos"] = {t_.origin_time: (np.array(t_.y_pred), np.array(t_.y_true)) for t_ in cv1.results[0].trials}
        origens = sorted(prev[REF])
        prev["Prophet com eventos"] = {}
        for o in origens:
            p = prophet_prev(s, o)
            prev["Prophet com eventos"][o] = (p, prev[REF][o][1])
        prev["Combinação (ETS, ARIMA com eventos, Prophet)"] = {
            o: (np.mean([prev[m][o][0] for m in ("ETS", "ARIMA com eventos", "Prophet com eventos")], axis=0), prev[REF][o][1])
            for o in origens}
        for m, dd in prev.items():
            for o in origens:
                linhas.append({"serie": nome_s, "modelo": m, "origem": f"{o:%Y-%m}", "wape": wape(*dd[o])})
        print(nome_s, "origens:", f"{origens[0]:%Y-%m} a {origens[-1]:%Y-%m}", len(origens), flush=True)

    val = pd.DataFrame(linhas)
    val.to_csv(saida / "validacao_origens.csv", index=False)
    resumo = []
    for nome_s, g in val.groupby("serie"):
        pivo = g.pivot(index="origem", columns="modelo", values="wape")
        for m in pivo.columns:
            dif = (pivo[m] - pivo[REF]).values
            lo, hi = ic_blocos(dif) if m != REF else (0.0, 0.0)
            resumo.append({"serie": nome_s, "modelo": m, "wape": float(pivo[m].mean()), "wape_pior_origem": float(pivo[m].max()),
                           "dif_vs_referencia": float(dif.mean()), "ic_inf": lo, "ic_sup": hi, "vence_referencia": float((pivo[m] < pivo[REF]).mean())})
    pd.DataFrame(resumo).to_csv(saida / "validacao_resumo.csv", index=False)
    print(pd.DataFrame(resumo).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    (saida / "resumo.json").write_text(json.dumps({
        "origens": ORIGENS, "linhas": int(len(d)), "municipios": int(d.fmun.nunique()), "inicio": f"{d.data.min():%Y-%m}", "fim": f"{d.data.max():%Y-%m}",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
