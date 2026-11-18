"""Risco de crédito, protocolo v2 (ver protocolo.md): variáveis derivadas, calibração e corte pelo resultado financeiro.

Dados: 45.000 pedidos de crédito de um conjunto anonimizado, dos quais 41.741 têm o desfecho conhecido. Ficam de fora
identificadores, telefone, e-mail, CEP, coordenadas, profissão, texto livre e o outro alvo (target_fraud).
Avaliação: validação cruzada estratificada de 5 dobras repetida 3 vezes, pré-processamento dentro de cada dobra,
probabilidades calibradas por regressão isotônica (25% do treino de cada dobra serve só à calibração).

Uso: python credito.py [--saida resultados]
Grava validacao_dobras.csv, resumo.csv, comparacao.csv, decisao.csv, balanceamento.csv e resumo.json em --saida.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
AQUI = Path(__file__).parent
SEMENTE = 42
NUMERICAS = ["score_3", "score_4", "score_5", "score_6", "risk_rate", "last_amount_borrowed", "last_borrowed_in_months",
             "credit_limit", "income", "reported_income", "ok_since", "n_bankruptcies", "n_defaulted_loans", "n_accounts",
             "n_issues", "application_time_in_funnel", "external_data_provider_credit_checks_last_month",
             "external_data_provider_credit_checks_last_year", "external_data_provider_email_seen_before",
             "external_data_provider_fraud_score", "hora_pedido"]
DERIVADAS = ["limite_sobre_renda", "ultimo_sobre_limite", "renda_declarada_sobre_renda", "calotes_por_conta",
             "scores_media", "scores_min", "campos_ausentes"]
CATEGORICAS = ["score_1", "score_2", "marketing_channel", "real_state", "facebook_profile"]
CUSTOS = (2, 4, 8)          # razão entre a perda de um mau pagador aprovado e o ganho de um bom pagador aprovado
RECUSA_FIXA = 0.20


def razao(a, b):
    return a / b.where(b > 0)


def carregar() -> pd.DataFrame:
    d = pd.read_csv(AQUI / "dados" / "acquisition_train.csv")
    d = d[d["target_default"].notna()].copy()
    d["y"] = d["target_default"].astype(str).str.lower().eq("true").astype(int)
    d["hora_pedido"] = pd.to_numeric(d["application_time_applied"].str[:2], errors="coerce")
    d[NUMERICAS] = d[NUMERICAS].replace([np.inf, -np.inf], np.nan)
    d["limite_sobre_renda"] = razao(d.credit_limit, d.income)
    d["ultimo_sobre_limite"] = razao(d.last_amount_borrowed, d.credit_limit)
    d["renda_declarada_sobre_renda"] = razao(d.reported_income, d.income)
    d["calotes_por_conta"] = razao(d.n_defaulted_loans, d.n_accounts)
    sc = d[["score_3", "score_4", "score_5", "score_6"]]
    d["scores_media"], d["scores_min"] = sc.mean(axis=1), sc.min(axis=1)
    d["campos_ausentes"] = d[NUMERICAS].isna().sum(axis=1)
    d[DERIVADAS] = d[DERIVADAS].replace([np.inf, -np.inf], np.nan)
    for c in CATEGORICAS:
        d[c] = d[c].astype("string").fillna("ausente")
    return d[NUMERICAS + DERIVADAS + CATEGORICAS + ["y"]].reset_index(drop=True)


def pre(numericas: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("esc", StandardScaler())]), numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    ])


def modelos() -> dict[str, tuple[Pipeline, list[str]]]:
    base, comb = NUMERICAS, NUMERICAS + DERIVADAS

    def pipe(clf, cols):
        return Pipeline([("pre", pre(cols)), ("clf", clf)]), cols

    lgbm = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, random_state=SEMENTE, verbose=-1)
    cat = dict(iterations=400, learning_rate=0.05, depth=5, random_seed=SEMENTE, verbose=0)
    return {
        "Regressão logística": pipe(LogisticRegression(max_iter=2000), base),
        "Regressão logística com derivadas": pipe(LogisticRegression(max_iter=2000), comb),
        "LightGBM": pipe(LGBMClassifier(**lgbm), base),
        "LightGBM com derivadas": pipe(LGBMClassifier(**lgbm), comb),
        "CatBoost": pipe(CatBoostClassifier(**cat), base),
        "CatBoost com derivadas": pipe(CatBoostClassifier(**cat), comb),
    }


def ks(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def resultado_por_pedido(y, aprovado, c: float) -> float:
    """Ganho 1 por bom pagador aprovado, perda c por mau pagador aprovado, dividido pelo número total de pedidos."""
    y = np.asarray(y)
    return float(((1 - y[aprovado]).sum() - c * y[aprovado].sum()) / len(y))


def ic_media(v, n: int = 2000, semente: int = SEMENTE) -> tuple[float, float]:
    rng = np.random.default_rng(semente)
    v = np.asarray(v, float)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def balanceamento(d: pd.DataFrame, saida: Path) -> None:
    """O erro de balancear a base antes de dividir: a minoria reamostrada com reposição aparece nos dois lados."""
    X, y = d.drop(columns="y"), d["y"]
    cols = NUMERICAS + CATEGORICAS

    def rf():
        return Pipeline([("pre", pre(NUMERICAS)), ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, class_weight="balanced_subsample", random_state=SEMENTE, n_jobs=-1))])

    X_tr, X_te, y_tr, y_te = train_test_split(X[cols], y, test_size=0.2, stratify=y, random_state=SEMENTE)
    base = pd.concat([X[cols], y], axis=1)
    mino, maio = base[base.y == 1], base[base.y == 0]
    mino_up = resample(mino, replace=True, n_samples=len(maio), random_state=SEMENTE)
    bal = pd.concat([maio, mino_up]).reset_index(drop=True)
    Xb, yb = bal.drop(columns="y"), bal["y"]
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.2, stratify=yb, random_state=SEMENTE)
    p_errado = rf().fit(Xb_tr, yb_tr).predict_proba(Xb_te)[:, 1]
    p_certo = rf().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    iguais = int(pd.concat([Xb_tr, Xb_te]).duplicated().sum())

    def met(yy, pp):
        return {"pr_auc": average_precision_score(yy, pp), "roc_auc": roc_auc_score(yy, pp), "ks": ks(yy, pp)}

    res = pd.DataFrame([
        {"protocolo": "balanceia antes de dividir", **met(yb_te, p_errado), "linhas_repetidas_entre_treino_e_teste": iguais},
        {"protocolo": "divide e só então ajusta", **met(y_te, p_certo), "linhas_repetidas_entre_treino_e_teste": 0},
    ])
    res.to_csv(saida / "balanceamento.csv", index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="resultados")
    ap.add_argument("--repeticoes", type=int, default=3)
    ap.add_argument("--so-balanceamento", action="store_true", help="roda só o experimento de balancear antes de dividir")
    args = ap.parse_args()
    saida = AQUI / args.saida
    saida.mkdir(exist_ok=True)

    d = carregar()
    if args.so_balanceamento:
        balanceamento(d, saida)
        return
    X, y = d.drop(columns="y"), d["y"]
    print(f"{len(d)} pedidos | inadimplência {y.mean():.1%} | {len(NUMERICAS)} numéricas, {len(DERIVADAS)} derivadas", flush=True)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeticoes, random_state=SEMENTE)
    linhas = []
    oof_cal: dict[str, list] = {m: [] for m in modelos()}          # (índices, p calibrada, repetição) para a decisão
    for k, (i_tr, i_te) in enumerate(rskf.split(X, y)):
        rep = k // 5
        i_fit, i_cal = train_test_split(i_tr, test_size=0.25, stratify=y.iloc[i_tr], random_state=SEMENTE + k)
        for nome, (pipe, cols) in modelos().items():
            pipe.fit(X.iloc[i_fit][cols + CATEGORICAS], y.iloc[i_fit])
            p_cal_set = pipe.predict_proba(X.iloc[i_cal][cols + CATEGORICAS])[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_cal_set, y.iloc[i_cal])
            p_raw = pipe.predict_proba(X.iloc[i_te][cols + CATEGORICAS])[:, 1]
            p = iso.predict(p_raw)
            yt = y.iloc[i_te]
            linhas.append({"modelo": nome, "dobra": k, "repeticao": rep, "pr_auc": average_precision_score(yt, p_raw),
                           "roc_auc": roc_auc_score(yt, p_raw), "ks": ks(yt, p_raw),
                           "brier_bruto": brier_score_loss(yt, p_raw), "brier_calibrado": brier_score_loss(yt, p)})
            oof_cal[nome].append((i_te, p, rep))
        print(f"dobra {k + 1}/{5 * args.repeticoes} pronta", flush=True)

    val = pd.DataFrame(linhas)
    val.to_csv(saida / "validacao_dobras.csv", index=False)
    resumo = val.groupby("modelo")[["pr_auc", "roc_auc", "ks", "brier_bruto", "brier_calibrado"]].mean().reset_index().sort_values("pr_auc", ascending=False)
    resumo.to_csv(saida / "resumo.csv", index=False)
    print(resumo.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # diferença pareada de PR-AUC contra a regressão logística original
    pivo = val.pivot(index="dobra", columns="modelo", values="pr_auc")
    comp = []
    for m in pivo.columns:
        if m == "Regressão logística":
            continue
        dif = (pivo[m] - pivo["Regressão logística"]).values
        lo, hi = ic_media(dif)
        comp.append({"modelo": m, "dif_pr_auc": float(dif.mean()), "ic_inf": lo, "ic_sup": hi, "vence": float((dif > 0).mean())})
    pd.DataFrame(comp).to_csv(saida / "comparacao.csv", index=False)
    print(pd.DataFrame(comp).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # decisão de crédito, com a probabilidade calibrada de cada pedido (média das repetições) e três razões de custo
    decisao = []
    for nome in modelos():
        p_medio = np.zeros(len(d))
        cont = np.zeros(len(d))
        for i_te, p, _ in oof_cal[nome]:
            p_medio[i_te] += p
            cont[i_te] += 1
        p_medio /= cont
        for c in CUSTOS:
            corte_otimo = 1 / (1 + c)
            aprov_otimo = p_medio < corte_otimo
            aprov_fixo = p_medio < np.quantile(p_medio, 1 - RECUSA_FIXA)
            decisao.append({"modelo": nome, "razao_custo": c,
                            "aprovar_todos": resultado_por_pedido(y, np.ones(len(d), bool), c),
                            "recusar_20pct": resultado_por_pedido(y, aprov_fixo, c),
                            "corte_otimo": resultado_por_pedido(y, aprov_otimo, c),
                            "recusados_no_corte_otimo": float(1 - aprov_otimo.mean())})
    dec = pd.DataFrame(decisao)
    dec.to_csv(saida / "decisao.csv", index=False)
    print(dec.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    balanceamento(d, saida)

    (saida / "resumo.json").write_text(json.dumps({
        "pedidos": int(len(d)), "inadimplencia": float(y.mean()), "dobras": int(5 * args.repeticoes), "custos": list(CUSTOS),
        "derivadas": DERIVADAS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
