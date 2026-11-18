"""Predição de evasão (churn) de clientes de telecomunicações, com protocolo sem vazamento.

Dados: Telco Customer Churn (amostra pública da IBM, 7.043 clientes, 21 colunas).
Protocolo: divisão estratificada 80/20 com o teste lacrado, pré-processamento dentro de um Pipeline
(ajustado só no treino de cada dobra), validação cruzada estratificada no treino para escolher o modelo,
e o limiar de decisão escolhido nas previsões fora da dobra, nunca no teste.

Uso: python churn.py [--saida resultados]
Grava resumo.json, validacao.csv, teste.csv, limiar.csv, taxa_por_grupo.csv e curvas.csv em --saida.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

AQUI = Path(__file__).parent
SEMENTE = 42
NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]
META_RECALL = 0.70      # fração dos clientes que vão evadir que a campanha de retenção precisa alcançar


def carregar() -> pd.DataFrame:
    d = pd.read_csv(AQUI / "dados" / "telco_churn.csv")
    # 11 clientes com tenure 0 têm TotalCharges em branco: ainda não pagaram nada, então o total é 0
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0.0)
    d["y"] = (d["Churn"] == "Yes").astype(int)
    return d.drop(columns=["customerID", "Churn"])


def preprocessador(X: pd.DataFrame) -> ColumnTransformer:
    cats = [c for c in X.columns if c not in NUMERICAS]
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
    ])


def modelos(X: pd.DataFrame) -> dict[str, Pipeline]:
    def pipe(clf):
        return Pipeline([("pre", preprocessador(X)), ("clf", clf)])
    return {
        "Prever que ninguém evade": pipe(DummyClassifier(strategy="prior")),
        "Regressão logística": pipe(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": pipe(RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                                                    random_state=SEMENTE, n_jobs=-1)),
        "Gradient boosting (scikit-learn)": pipe(HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                                                random_state=SEMENTE)),
        "LightGBM": pipe(LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                        random_state=SEMENTE, verbose=-1)),
        "CatBoost": pipe(CatBoostClassifier(iterations=400, learning_rate=0.05, depth=4, random_seed=SEMENTE, verbose=0)),
    }


def modelo_tabpfn(X: pd.DataFrame) -> Pipeline | None:
    """TabPFN (modelo pré-treinado para dados tabulares). Exige aceitar a licença e definir TABPFN_TOKEN."""
    if not os.environ.get("TABPFN_TOKEN"):
        return None
    from tabpfn import TabPFNClassifier
    cats = [c for c in X.columns if c not in NUMERICAS]
    pre = ColumnTransformer([("num", StandardScaler(), NUMERICAS),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)])
    return Pipeline([("pre", pre), ("clf", TabPFNClassifier(device="cpu", n_estimators=4, ignore_pretraining_limits=True))])


def limiar_para_recall(y, p, meta: float) -> float:
    """Maior limiar cujo recall ainda alcança a meta (quanto maior o limiar, menos clientes abordados)."""
    prec, rec, lim = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= meta)[0]
    return float(lim[ok[-1]])


def metricas_no_limiar(y, p, lim: float) -> dict:
    prev = p >= lim
    tp = int(((prev) & (y == 1)).sum())
    return {
        "limiar": lim,
        "abordados": float(prev.mean()),
        "recall": tp / int((y == 1).sum()),
        "precisao": tp / max(int(prev.sum()), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="resultados")
    args = ap.parse_args()
    saida = AQUI / args.saida
    saida.mkdir(exist_ok=True)

    d = carregar()
    X, y = d.drop(columns="y"), d["y"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMENTE)
    print(f"{len(d)} clientes | evasão {y.mean():.1%} | treino {len(X_tr)} | teste lacrado {len(X_te)}")

    # taxa de evasão por contrato e por faixa de tempo de casa (só descrição, no conjunto todo)
    faixa = pd.cut(X["tenure"], [-1, 6, 12, 24, 48, 100], labels=["0 a 6", "7 a 12", "13 a 24", "25 a 48", "49 a 72"])
    pd.concat([
        d.groupby("Contract").y.agg(["mean", "size"]).rename_axis("grupo").assign(tipo="contrato"),
        d.groupby(faixa, observed=True).y.agg(["mean", "size"]).rename_axis("grupo").assign(tipo="meses de casa"),
    ]).reset_index().to_csv(saida / "taxa_por_grupo.csv", index=False)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEMENTE)
    linhas, oof = [], {}
    todos = modelos(X_tr)
    if (tb := modelo_tabpfn(X_tr)) is not None:
        todos["TabPFN"] = tb
    for nome, pipe in todos.items():
        p = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        oof[nome] = p
        linhas.append({"modelo": nome, "roc_auc": roc_auc_score(y_tr, p), "pr_auc": average_precision_score(y_tr, p)})
    val = pd.DataFrame(linhas)
    val.to_csv(saida / "validacao.csv", index=False)
    print(val.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # o modelo escolhido é o de maior PR-AUC na validação cruzada, entre os que não são a referência
    escolhido = val[val.modelo != "Prever que ninguém evade"].sort_values("pr_auc").iloc[-1].modelo
    print("escolhido:", escolhido)

    lim = limiar_para_recall(y_tr, oof[escolhido], META_RECALL)
    pipe = todos[escolhido].fit(X_tr, y_tr)
    p_te = pipe.predict_proba(X_te)[:, 1]
    teste = {
        "modelo": escolhido,
        "roc_auc": roc_auc_score(y_te, p_te),
        "pr_auc": average_precision_score(y_te, p_te),
        "taxa_base": float(y_te.mean()),
        **metricas_no_limiar(y_te, p_te, lim),
    }
    # referência no teste: abordar clientes ao acaso na mesma proporção
    teste["precisao_ao_acaso"] = teste["taxa_base"]
    pd.DataFrame([teste]).to_csv(saida / "teste.csv", index=False)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in teste.items()})

    # compromisso entre recall e precisão no teste, por limiar
    prec, rec, ls = precision_recall_curve(y_te, p_te)
    pd.DataFrame({"limiar": np.append(ls, 1.0), "precisao": prec, "recall": rec}).to_csv(saida / "curvas.csv", index=False)
    tab = []
    for r in (0.5, 0.6, 0.7, 0.8, 0.9):
        l = limiar_para_recall(y_tr, oof[escolhido], r)
        tab.append({"recall_meta_no_treino": r, **metricas_no_limiar(y_te, p_te, l)})
    pd.DataFrame(tab).to_csv(saida / "limiar.csv", index=False)

    # o que importa para o modelo escolhido: queda de PR-AUC no teste ao embaralhar cada coluna
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(pipe, X_te, y_te, scoring="average_precision", n_repeats=20, random_state=SEMENTE, n_jobs=-1)
    pd.DataFrame({"coluna": X_te.columns, "queda_pr_auc": imp.importances_mean}).sort_values("queda_pr_auc", ascending=False) \
        .to_csv(saida / "importancia.csv", index=False)
    (saida / "resumo.json").write_text(json.dumps({
        "clientes": int(len(d)), "evasao": float(y.mean()), "treino": int(len(X_tr)), "teste": int(len(X_te)),
        "escolhido": escolhido, "meta_recall": META_RECALL, **{k: v for k, v in teste.items() if k != "modelo"},
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
