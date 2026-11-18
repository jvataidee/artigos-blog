"""Airbnb em Toronto, recorte de setembro de 2026: exploração dos preços e um modelo de preço com divisão espacial.

Dados: listings.csv.gz do Inside Airbnb (coletado em 17 e 18/09/2026), 22.050 anúncios. Só anúncios com preço
entram (16.760); preços acima de 2.000 dólares canadenses saem como valores extremos.
Modelo: log do preço, em validação cruzada por blocos espaciais (GroupKFold) no treino, e teste lacrado em 20% dos
blocos da grade. Compara com uma divisão aleatória, em que anúncios do mesmo prédio caem nos dois lados.

Uso: python airbnb.py [--saida resultados]
Grava resumo.json, preco_por_tipo.csv, bairros.csv, anfitriao.csv, validacao.csv, teste.csv, divisao.csv e importancia.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier  # noqa: F401  (mantém o mesmo ambiente dos outros estudos)
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

AQUI = Path(__file__).parent
SEMENTE = 42
PRECO_MAX = 2000
CENTRO = (43.6453, -79.3806)           # Union Station, referência do centro
CELULA = 0.015                         # lado da célula da grade, em graus (cerca de 1,5 km)
NUMERICAS = ["accommodates", "bedrooms", "beds", "minimum_nights", "availability_30", "number_of_reviews_ltm",
             "review_scores_rating", "host_listings_count", "amenidades", "anos_anfitriao", "dist_centro_km",
             "latitude", "longitude"]
CATEGORICAS = ["room_type", "host_is_superhost", "instant_bookable", "neighbourhood_cleansed"]


def km(lat, lon):
    """Distância aproximada ao centro, em km (plano local, suficiente para uma cidade)."""
    return np.hypot((lat - CENTRO[0]) * 111.0, (lon - CENTRO[1]) * 111.0 * np.cos(np.radians(CENTRO[0])))


def carregar() -> pd.DataFrame:
    d = pd.read_csv(AQUI / "dados" / "listings.csv.gz")
    d["preco"] = d["price"].str.replace(r"[$,]", "", regex=True).astype(float)
    d["amenidades"] = d["amenities"].str.count(",") + 1
    d["anos_anfitriao"] = (pd.Timestamp("2026-09-18") - pd.to_datetime(d["host_since"])).dt.days / 365.25
    d["dist_centro_km"] = km(d["latitude"], d["longitude"])
    d["bloco"] = (np.floor(d["latitude"] / CELULA).astype(int).astype(str) + "_" + np.floor(d["longitude"] / CELULA).astype(int).astype(str))
    return d


def preprocessador() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("esc", StandardScaler())]), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAS),
    ])


def modelos() -> dict[str, Pipeline]:
    def pipe(reg):
        return Pipeline([("pre", preprocessador()), ("reg", reg)])
    return {
        "Regressão linear (Ridge)": pipe(Ridge(alpha=3.0)),
        "LightGBM": pipe(LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEMENTE, verbose=-1)),
        "CatBoost": pipe(CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, random_seed=SEMENTE, verbose=0)),
    }


def metricas(y_log, p_log) -> dict:
    y, p = np.expm1(y_log), np.expm1(p_log)
    return {"mae_usd": float(mean_absolute_error(y, p)), "erro_mediano_pct": float(np.median(np.abs(y - p) / y)),
            "r2_log": float(r2_score(y_log, p_log))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="resultados")
    args = ap.parse_args()
    saida = AQUI / args.saida
    saida.mkdir(exist_ok=True)

    bruto = carregar()
    d = bruto[bruto.preco.notna() & (bruto.preco <= PRECO_MAX) & bruto.bedrooms.notna()].reset_index(drop=True)
    print(f"{len(bruto)} anúncios | com preço {int(bruto.preco.notna().sum())} | usados {len(d)} | blocos {d.bloco.nunique()}")

    # exploração
    tipo = d.groupby("room_type").preco.agg(n="size", mediana="median", q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75)).reset_index()
    tipo.to_csv(saida / "preco_por_tipo.csv", index=False)
    casa = d[d.room_type == "Entire home/apt"]
    bai = casa.groupby("neighbourhood_cleansed").preco.agg(n="size", mediana="median").query("n >= 40").sort_values("mediana").reset_index()
    pd.concat([bai.head(8).assign(grupo="mais baratos"), bai.tail(8).assign(grupo="mais caros")]).to_csv(saida / "bairros.csv", index=False)
    multi = (bruto.calculated_host_listings_count > 1)
    pd.DataFrame([{
        "anuncios": len(bruto), "de_anfitrioes_com_varios": float(multi.mean()), "sem_avaliacao": float((bruto.number_of_reviews == 0).mean()),
        "superhost": float((bruto.host_is_superhost == "t").mean()),
        "corr_preco_dist_centro": float(np.corrcoef(np.log(casa.preco), casa.dist_centro_km)[0, 1]),
    }]).to_csv(saida / "anfitriao.csv", index=False)

    # divisão em blocos: 20% dos blocos da grade viram teste lacrado
    y = np.log1p(d.preco)
    X = d[NUMERICAS + CATEGORICAS]
    rng = np.random.default_rng(SEMENTE)
    blocos = d.bloco.unique()
    teste_b = set(rng.choice(blocos, size=int(len(blocos) * 0.2), replace=False))
    m_te = d.bloco.isin(teste_b).to_numpy()
    X_tr, y_tr, g_tr = X[~m_te], y[~m_te], d.bloco[~m_te]
    X_te, y_te = X[m_te], y[m_te]
    print(f"treino {len(X_tr)} ({g_tr.nunique()} blocos) | teste lacrado {len(X_te)} ({len(teste_b)} blocos)")

    # referência: mediana do preço por tipo de quarto e número de hóspedes
    med = d[~m_te].groupby(["room_type", "accommodates"]).preco.median()
    fallback = d[~m_te].groupby("room_type").preco.median()

    def ref(df):
        return np.log1p([med.get((r, a), fallback[r]) for r, a in zip(df.room_type, df.accommodates)])

    cv = GroupKFold(5)
    linhas = []
    oof = {}
    for nome, pipe in modelos().items():
        p = cross_val_predict(pipe, X_tr, y_tr, cv=cv, groups=g_tr)
        oof[nome] = p
        linhas.append({"modelo": nome, **metricas(y_tr, p)})
    linhas.insert(0, {"modelo": "Mediana por tipo e hóspedes", **metricas(y_tr, ref(d[~m_te]))})
    val = pd.DataFrame(linhas)
    val.to_csv(saida / "validacao.csv", index=False)
    print(val.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    escolhido = val[val.modelo != "Mediana por tipo e hóspedes"].sort_values("mae_usd").iloc[0].modelo
    print("escolhido:", escolhido)
    pipe = modelos()[escolhido].fit(X_tr, y_tr)
    p_te = pipe.predict(X_te)
    teste = pd.DataFrame([
        {"modelo": "Mediana por tipo e hóspedes", **metricas(y_te, ref(d[m_te]))},
        {"modelo": escolhido, **metricas(y_te, p_te)},
    ])
    teste.to_csv(saida / "teste.csv", index=False)
    print(teste.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # a mesma divisão sem blocos: anúncios do mesmo prédio e da mesma rua caem nos dois lados
    linhas = []
    for semente in range(5):
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=semente)
        pa = modelos()[escolhido].fit(Xa, ya).predict(Xb)
        linhas.append({"divisao": f"aleatória {semente}", **metricas(yb, pa)})
    linhas.append({"divisao": "por blocos", **metricas(y_te, p_te)})
    comp = pd.DataFrame(linhas)
    comp.to_csv(saida / "divisao.csv", index=False)
    print(comp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    imp = permutation_importance(pipe, X_te, y_te, scoring="neg_mean_absolute_error", n_repeats=5, random_state=SEMENTE, n_jobs=-1)
    pd.DataFrame({"coluna": X_te.columns, "aumento_do_erro": imp.importances_mean}).sort_values("aumento_do_erro", ascending=False) \
        .to_csv(saida / "importancia.csv", index=False)
    (saida / "resumo.json").write_text(json.dumps({
        "anuncios": int(len(bruto)), "com_preco": int(bruto.preco.notna().sum()), "usados": int(len(d)), "blocos": int(d.bloco.nunique()),
        "treino": int(len(X_tr)), "teste": int(len(X_te)), "escolhido": escolhido, "coleta": "2026-09-17 e 2026-09-18",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
