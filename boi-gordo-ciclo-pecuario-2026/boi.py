"""Boi gordo e ciclo pecuário: a participação de fêmeas no abate ajuda a prever o preço real do boi?

Hipótese (ciclo pecuário). Quando o produtor descarta mais fêmeas, o abate de fêmeas sobe e o rebanho de matrizes
encolhe. Se isso antecede o preço do boi, a participação de fêmeas no abate, defasada, melhora a previsão do preço
real um ano à frente (4 trimestres) em relação a modelos que só olham o próprio preço.

Dados (todos abertos, ver baixar_dados.py): preço do boi gordo no Paraná (IPEADATA), abates por tipo de rebanho
(IBGE/SIDRA, trimestral), IPCA e dólar (Banco Central). O preço é deflacionado pelo IPCA e vira média trimestral, em log.
Protocolo: AutoHybTS com fh=4; os últimos 4 trimestres ficam lacrados (teste); a comparação usa validação cruzada
com origem móvel nos trimestres anteriores. Modelos: passeio aleatório sazonal, ARIMA e ETS (só o preço) e ARIMA
com a participação de fêmeas e o dólar defasados (exógenas).

Uso: python boi.py [--saida resultados]
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).parent
FH = 4


def carregar() -> pd.DataFrame:
    d = AQUI / "dados"
    preco = pd.read_csv(d / "preco_boi_pr.csv", parse_dates=["data"])
    preco = preco[preco.data >= "1995-01-01"].set_index("data").preco_arroba
    ipca = pd.read_csv(d / "ipca_mensal.csv", parse_dates=["data"]).set_index("data").ipca_var
    indice = (1 + ipca / 100).cumprod()
    real = preco / indice.reindex(preco.index) * indice.iloc[-1]           # R$ por arroba, a preços de out/2026
    dolar = pd.read_csv(d / "dolar_mensal.csv", parse_dates=["data"]).set_index("data").dolar
    ab = pd.read_csv(d / "abates_trimestre.csv")
    ab["trim"] = pd.PeriodIndex([f"{int(str(t)[:4])}Q{int(str(t)[4:])}" for t in ab.trimestre], freq="Q")
    ab = ab.set_index("trim")
    base = ab[["Bois", "Vacas", "Novilhos", "Novilhas"]].sum(axis=1)
    out = pd.DataFrame({
        "preco_real": real.groupby(real.index.to_period("Q")).mean(),
        "preco_nominal": preco.groupby(preco.index.to_period("Q")).mean(),
        "dolar": dolar.groupby(dolar.index.to_period("Q")).mean(),
        "femeas": (ab.Vacas + ab.Novilhas) / base,
        "abates": ab.Total,
    })
    # só trimestres completos de preço e de abate
    out = out.dropna(subset=["femeas", "preco_real"])
    out = out[out.index <= pd.Period("2026Q2")]
    out["ln_preco"] = np.log(out.preco_real)
    out["ln_dolar"] = np.log(out.dolar)
    return out


def serie_e_exog(q: pd.DataFrame, cols: list[str]):
    y = q.ln_preco.copy()
    y.index = pd.DatetimeIndex(q.index.to_timestamp(), freq="QS")
    X = None
    if cols:
        X = q[cols].copy()
        X.index = y.index
    return y, X


ESPEC = {
    "Só o preço": ([], None),
    "Com fêmeas": (["femeas"], {"femeas": [4, 8, 12]}),
    "Com dólar": (["ln_dolar"], {"ln_dolar": [4]}),
    "Com fêmeas e dólar": (["femeas", "ln_dolar"], {"femeas": [4, 8, 12], "ln_dolar": [4]}),
}


def boot_ic(dif: np.ndarray, n: int = 2000, semente: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(semente)
    m = [rng.choice(dif, len(dif)).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    from autohybts import AutoHybridExperiment

    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="resultados")
    ap.add_argument("--origens", type=int, default=40)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    saida = AQUI / args.saida
    saida.mkdir(exist_ok=True)

    q = carregar()
    q.to_csv(saida / "serie_trimestral.csv")
    print(f"{len(q)} trimestres de {q.index.min()} a {q.index.max()}")

    # leitura exploratória: correlação da variação anual do preço com a participação de fêmeas defasada
    var4 = q.ln_preco.diff(4)
    cc = pd.DataFrame({"defasagem": range(0, 17), "correlacao": [var4.corr(q.femeas.shift(k)) for k in range(0, 17)]})
    cc.to_csv(saida / "correlacao_cruzada.csv", index=False)

    linhas_val, linhas_te = [], []
    for nome, (cols, lags) in ESPEC.items():
        y, X = serie_e_exog(q, cols)
        exp = AutoHybridExperiment(fh=FH, session_id=42, exog_lags=lags or {}).fit(y, X)
        if X is None:
            modelos = {"seasonal_naive": "Referência sazonal", "arima": "ARIMA só o preço", "exponential_smoothing": "ETS só o preço"}
            specs = list(modelos)
            rotulo = dict(modelos)
        else:
            specs = [exp.create_model("arima", exog=True)]
            rotulo = {}
        cv = exp.cross_validate(models=specs, folds=args.origens, step=1)
        origens = cv.splits.origin.tolist()
        for r in cv.results:
            base = r.candidate.base.name
            etiqueta = rotulo.get(base, f"ARIMA {nome.lower()}")
            for (fid, row), o in zip(r.to_frame().iterrows(), origens):
                linhas_val.append({"modelo": etiqueta, "origem": str(o)[:10], "mae_ln": float(row.MAE)})
        real_te = y.iloc[-FH:]
        for s in (list(rotulo) if X is None else ["arima"]):
            fin = exp.finalize_model(exp.create_model(s, exog=X is not None))
            mae = float(exp.evaluate_external(fin).metrics["MAE"].iloc[0])
            linhas_te.append({"modelo": rotulo.get(s, f"ARIMA {nome.lower()}"), "mae_ln": mae})
        print(nome, "ok", flush=True)

    val = pd.DataFrame(linhas_val)
    te = pd.DataFrame(linhas_te)
    val.to_csv(saida / "validacao_origens.csv", index=False)
    te.to_csv(saida / "teste.csv", index=False)

    pivo = val.pivot_table(index="origem", columns="modelo", values="mae_ln")
    base = pivo["ARIMA só o preço"]
    resumo = []
    for m in pivo.columns:
        dif = (pivo[m] - base).dropna().values
        lo, hi = boot_ic(dif) if m != "ARIMA só o preço" else (0.0, 0.0)
        resumo.append({"modelo": m, "mae_ln": float(pivo[m].mean()), "dif_vs_arima": float(dif.mean()), "ic_inf": lo, "ic_sup": hi,
                       "vence_arima": float((pivo[m] < base).mean())})
    pd.DataFrame(resumo).sort_values("mae_ln").to_csv(saida / "validacao_resumo.csv", index=False)
    (saida / "resumo.json").write_text(json.dumps({
        "trimestres": int(len(q)), "inicio": str(q.index.min()), "fim": str(q.index.max()), "origens": int(pivo.shape[0]),
        "teste_inicio": str(q.index[-FH]), "teste_fim": str(q.index[-1]),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(resumo).sort_values("mae_ln").to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(te.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
