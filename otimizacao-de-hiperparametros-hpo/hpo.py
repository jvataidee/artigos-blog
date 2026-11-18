"""Otimização de hiperparâmetros: busca aleatória, TPE, CMA-ES e um algoritmo genético, com o mesmo orçamento.

Tarefa: ajustar 7 hiperparâmetros de um LightGBM que prevê a evasão de clientes (base Telco, 7.043 clientes).
Objetivo: PR-AUC em validação cruzada estratificada de 3 dobras, só no treino (80%). O teste lacrado (20%) só
avalia a melhor configuração de cada execução, depois que a busca termina.
Cada método recebe 60 avaliações do objetivo e roda com 10 sementes. A configuração padrão do LightGBM
entra como referência.

Uso: python hpo.py [--sementes 10] [--avaliacoes 60] [--saida resultados]
Grava convergencia.csv, melhores.csv, padrao.csv e resumo.json em --saida.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

AQUI = Path(__file__).parent
SEMENTE_DADOS = 42
# nome: (tipo, mínimo, máximo, escala logarítmica)
ESPACO = {
    "n_estimators": ("int", 50, 400, False),
    "learning_rate": ("float", 0.005, 0.3, True),
    "num_leaves": ("int", 4, 96, False),
    "min_child_samples": ("int", 5, 100, False),
    "subsample": ("float", 0.5, 1.0, False),
    "colsample_bytree": ("float", 0.4, 1.0, False),
    "reg_lambda": ("float", 0.001, 10.0, True),
}
NOMES = list(ESPACO)


def carregar():
    d = pd.read_csv(AQUI / "dados" / "telco_churn.csv")
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0.0)
    y = (d["Churn"] == "Yes").astype(int)
    X = pd.get_dummies(d.drop(columns=["customerID", "Churn"]), dtype=float)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMENTE_DADOS)


def de_unitario(u: np.ndarray) -> dict:
    """Converte um vetor em [0, 1]^7 nos hiperparâmetros reais."""
    p = {}
    for x, (nome, (tipo, lo, hi, log)) in zip(np.clip(u, 0, 1), ESPACO.items()):
        v = np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))) if log else lo + x * (hi - lo)
        p[nome] = int(round(v)) if tipo == "int" else float(v)
    return p


def modelo(p: dict, semente: int = 0) -> LGBMClassifier:
    return LGBMClassifier(**p, subsample_freq=1, n_jobs=4, verbose=-1, random_state=semente)


class Objetivo:
    def __init__(self, X, y):
        self.X, self.y = X, y
        self.cv = StratifiedKFold(3, shuffle=True, random_state=0)
        self.n = 0

    def __call__(self, p: dict) -> float:
        self.n += 1
        prob = cross_val_predict(modelo(p), self.X, self.y, cv=self.cv, method="predict_proba")[:, 1]
        return float(average_precision_score(self.y, prob))


# ------------------------------------------------------------- os quatro métodos
def aleatoria(obj, n, semente):
    rng = np.random.default_rng(semente)
    hist = []
    for _ in range(n):
        p = de_unitario(rng.random(len(NOMES)))
        hist.append((p, obj(p)))
    return hist


def _optuna(obj, n, sampler):
    estudo = optuna.create_study(direction="maximize", sampler=sampler)
    hist = []

    def f(t):
        p = {}
        for nome, (tipo, lo, hi, log) in ESPACO.items():
            p[nome] = t.suggest_int(nome, lo, hi) if tipo == "int" else t.suggest_float(nome, lo, hi, log=log)
        s = obj(p)
        hist.append((p, s))
        return s

    estudo.optimize(f, n_trials=n)
    return hist


def tpe(obj, n, semente):
    return _optuna(obj, n, optuna.samplers.TPESampler(seed=semente, n_startup_trials=10))


def cma(obj, n, semente):
    return _optuna(obj, n, optuna.samplers.CmaEsSampler(seed=semente, n_startup_trials=10))


def genetico(obj, n, semente, populacao=10, torneio=3, p_cruz=0.8, alfa=0.3, p_mut=0.2, sigma=0.12):
    """Algoritmo genético com genes reais em [0, 1], seleção por torneio, cruzamento BLX-alfa, mutação gaussiana
    e sobrevivência (mu + lambda): os melhores entre pais e filhos formam a geração seguinte."""
    rng = np.random.default_rng(semente)
    d = len(NOMES)
    hist: list = []

    def avaliar(u):
        p = de_unitario(u)
        s = obj(p)
        hist.append((p, s))
        return s

    pop = [rng.random(d) for _ in range(populacao)]
    fit = [avaliar(u) for u in pop]
    while len(hist) < n:
        filhos = []
        while len(filhos) < populacao and len(hist) + len(filhos) < n:
            def escolher():
                idx = rng.choice(len(pop), torneio, replace=False)
                return pop[max(idx, key=lambda i: fit[i])]
            a, b = escolher(), escolher()
            if rng.random() < p_cruz:
                lo, hi = np.minimum(a, b), np.maximum(a, b)
                amp = hi - lo
                filho = rng.uniform(lo - alfa * amp, hi + alfa * amp)
            else:
                filho = a.copy()
            mut = rng.random(d) < p_mut
            filho = np.clip(filho + mut * rng.normal(0, sigma, d), 0, 1)
            filhos.append(filho)
        fit_f = [avaliar(u) for u in filhos]
        todos = pop + filhos
        fits = fit + fit_f
        ordem = np.argsort(fits)[::-1][:populacao]
        pop, fit = [todos[i] for i in ordem], [fits[i] for i in ordem]
    return hist


METODOS = {"Busca aleatória": aleatoria, "TPE (Optuna)": tpe, "CMA-ES (Optuna)": cma, "Algoritmo genético": genetico}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementes", type=int, default=10)
    ap.add_argument("--avaliacoes", type=int, default=60)
    ap.add_argument("--saida", default="resultados")
    args = ap.parse_args()
    saida = AQUI / args.saida
    saida.mkdir(exist_ok=True)

    X_tr, X_te, y_tr, y_te = carregar()
    print(f"treino {len(X_tr)} | teste lacrado {len(X_te)} | {len(NOMES)} hiperparâmetros | {args.avaliacoes} avaliações por execução")

    obj0 = Objetivo(X_tr, y_tr)
    padrao = {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 20,
              "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 0.0}
    cv_padrao = obj0(padrao)
    te_padrao = average_precision_score(y_te, modelo(padrao).fit(X_tr, y_tr).predict_proba(X_te)[:, 1])
    pd.DataFrame([{"cv_pr_auc": cv_padrao, "teste_pr_auc": te_padrao, **padrao}]).to_csv(saida / "padrao.csv", index=False)
    print(f"padrão do LightGBM: CV {cv_padrao:.4f} | teste {te_padrao:.4f}")

    conv, melhores = [], []
    t0 = time.time()
    for semente in range(args.sementes):
        for nome, fn in METODOS.items():
            obj = Objetivo(X_tr, y_tr)
            hist = fn(obj, args.avaliacoes, semente)
            scores = [s for _, s in hist][: args.avaliacoes]
            melhor_ate = np.maximum.accumulate(scores)
            conv += [{"metodo": nome, "semente": semente, "avaliacao": i + 1, "melhor_ate_aqui": float(b)} for i, b in enumerate(melhor_ate)]
            p_best = hist[int(np.argmax(scores))][0]
            te = average_precision_score(y_te, modelo(p_best).fit(X_tr, y_tr).predict_proba(X_te)[:, 1])
            melhores.append({"metodo": nome, "semente": semente, "cv_pr_auc": float(max(scores)), "teste_pr_auc": float(te), **p_best})
        print(f"semente {semente} pronta | {time.time() - t0:.0f} s", flush=True)
        pd.DataFrame(conv).to_csv(saida / "convergencia.csv", index=False)
        pd.DataFrame(melhores).to_csv(saida / "melhores.csv", index=False)

    m = pd.DataFrame(melhores)
    res = m.groupby("metodo")[["cv_pr_auc", "teste_pr_auc"]].agg(["mean", "std"]).round(4)
    print(res.to_string())
    (saida / "resumo.json").write_text(json.dumps({
        "treino": len(X_tr), "teste": len(X_te), "hiperparametros": len(NOMES), "avaliacoes": args.avaliacoes,
        "sementes": args.sementes, "cv_padrao": cv_padrao, "teste_padrao": float(te_padrao),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
