"""Detecção de fraude em cartões de crédito, protocolo v2 (ver protocolo.md): quatro janelas de tempo e valor da transação.

Treino sempre no passado, avaliação em janelas de 6 horas (24 a 30 h, 30 a 36 h, 36 a 42 h e 42 a 48 h), com
orçamento de 0,2% de alertas por janela. Compara quatro modelos (com e sem a coluna Time) e duas regras de ranking (probabilidade, e
probabilidade vezes o valor da transação). Métricas: PR-AUC, recall no orçamento e valor recuperado.

Uso: python fraude.py [--saida resultados]
Grava janelas.csv, resumo_modelos.csv, comparacao.csv e resumo.json em --saida.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AQUI = Path(__file__).parent
SEMENTE = 42
ORCAMENTO = 0.002        # fração das transações que a equipe consegue investigar em cada janela
JANELAS = [(24, 30), (30, 36), (36, 42), (42, 48)]
REGRAS = ("probabilidade", "probabilidade x valor")


def carregar() -> tuple[pd.DataFrame, np.ndarray]:
    bruto = pd.read_csv(AQUI / "dados" / "creditcard.csv").sort_values("Time").reset_index(drop=True)
    valor = bruto["Amount"].to_numpy()                   # valor em dólares, sem transformação
    d = bruto.copy()
    d["Time"] = d["Time"] / 3600.0                       # horas desde a primeira transação
    d["Amount"] = np.log1p(d["Amount"])
    return d, valor


def modelos(escala_pos: float) -> dict:
    return {
        "Regressão logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                                random_state=SEMENTE, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   scale_pos_weight=escala_pos, random_state=SEMENTE, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, random_seed=SEMENTE, verbose=0,
                                       auto_class_weights="SqrtBalanced"),
    }


def ic_media(v, n: int = 2000, semente: int = SEMENTE) -> tuple[float, float]:
    """Intervalo de 95% da média, reamostrando as fraudes com reposição."""
    rng = np.random.default_rng(semente)
    v = np.asarray(v, float)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="resultados")
    args = ap.parse_args()
    saida = AQUI / args.saida
    saida.mkdir(exist_ok=True)

    d, valor = carregar()
    X, y = d.drop(columns="Class"), d["Class"]
    print(f"{len(d)} transações, {int(y.sum())} fraudes ({y.mean():.3%})")
    # emenda 1 do protocolo: a coluna Time (horas desde o início) sai do intervalo do treino nas janelas futuras;
    # a variante troca Time pela hora do dia
    X_sem_time = X.drop(columns="Time").assign(hora_dia=(X.Time % 24))

    linhas = []
    achadas: dict[tuple[str, str], list] = {}      # (modelo, regra) -> 1 se a fraude foi achada, por fraude, nas 4 janelas
    for h0, h1 in JANELAS:
        tr = (d.Time < h0).to_numpy()
        te = ((d.Time >= h0) & (d.Time < h1)).to_numpy()
        ytr, yv = y[tr], y[te].to_numpy()
        pos = float((ytr == 0).sum() / max(int((ytr == 1).sum()), 1))
        k = max(int(round(te.sum() * ORCAMENTO)), 1)
        print(f"janela {h0}-{h1} h: treino {int(tr.sum())} ({int(ytr.sum())} fraudes), avaliação {int(te.sum())} "
              f"({int(yv.sum())} fraudes), {k} alertas", flush=True)
        for nome, m, Xv in [(n, mm, X) for n, mm in modelos(pos).items()] + [(n + " [sem Time]", mm, X_sem_time) for n, mm in modelos(pos).items()]:
            p = m.fit(Xv[tr], ytr).predict_proba(Xv[te])[:, 1]
            for regra, escore in zip(REGRAS, (p, p * valor[te])):
                hit = np.zeros(te.sum(), bool)
                hit[np.argsort(-escore)[:k]] = True
                tp = int((hit & (yv == 1)).sum())
                v_tot = float(valor[te][yv == 1].sum())
                v_ach = float(valor[te][hit & (yv == 1)].sum())
                linhas.append({"janela": f"{h0}-{h1} h", "modelo": nome, "regra": regra, "fraudes": int(yv.sum()), "alertas": k,
                               "pr_auc": float(average_precision_score(yv, p)), "recall": tp / int(yv.sum()), "precisao": tp / k,
                               "valor_recuperado": v_ach / v_tot if v_tot else float("nan")})
                achadas.setdefault((nome, regra), []).extend(hit[yv == 1].astype(int).tolist())

    jan = pd.DataFrame(linhas)
    jan.to_csv(saida / "janelas.csv", index=False)

    res = jan.groupby(["modelo", "regra"]).agg(pr_auc=("pr_auc", "mean"), precisao=("precisao", "mean"),
                                               valor_recuperado=("valor_recuperado", "mean")).reset_index()
    ic = []
    for (m, r), v in achadas.items():
        lo, hi = ic_media(v)
        ic.append({"modelo": m, "regra": r, "recall_agregado": float(np.mean(v)), "ic_inf": lo, "ic_sup": hi})
    res = res.merge(pd.DataFrame(ic), on=["modelo", "regra"]).sort_values("recall_agregado", ascending=False)
    res.to_csv(saida / "resumo_modelos.csv", index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # diferença pareada (por fraude) de cada combinação contra a regressão logística por probabilidade
    base = np.asarray(achadas[("Regressão logística", "probabilidade")])
    comp = []
    for (m, r), v in achadas.items():
        if (m, r) == ("Regressão logística", "probabilidade"):
            continue
        dif = np.asarray(v) - base
        lo, hi = ic_media(dif)
        comp.append({"modelo": m, "regra": r, "dif_recall": float(dif.mean()), "ic_inf": lo, "ic_sup": hi})
    pd.DataFrame(comp).to_csv(saida / "comparacao.csv", index=False)
    print(pd.DataFrame(comp).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    (saida / "resumo.json").write_text(json.dumps({
        "transacoes": int(len(d)), "fraudes": int(y.sum()), "taxa_fraude": float(y.mean()),
        "janelas": [f"{a} a {b} h" for a, b in JANELAS], "fraudes_nas_janelas": int(jan.drop_duplicates("janela").fraudes.sum()),
        "orcamento": ORCAMENTO,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
