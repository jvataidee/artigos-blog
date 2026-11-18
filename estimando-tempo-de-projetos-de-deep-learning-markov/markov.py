"""Tempo de um projeto de deep learning como cadeia de Markov absorvente.

Estados: Amostra, Treino, Validação, Produção (absorvente). Cada transição dura 1 semana.
Calcula a resposta exata (matriz fundamental e potências da matriz) e a confere por simulação.

Uso: python markov.py [--simulacoes 10000] [--saida resultados]
Grava resumo.json, distribuicao.csv, simulacao.csv e cenarios.csv em --saida.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ESTADOS = ["Amostra", "Treino", "Validação", "Produção"]
# Probabilidades definidas em conversa com cientistas de dados (julgamento, não medição).
P = np.array([
    [0.4, 0.6, 0.0, 0.0],
    [0.2, 0.5, 0.3, 0.0],
    [0.3, 0.4, 0.0, 0.3],
    [0.3, 0.5, 0.2, 0.0],   # da produção só se sai voltando; no estudo, chegar aqui encerra o projeto
])
SEMANA_MIN, SEMANA_MAX = 4, 16
MAX_SEMANAS = 600


def absorvente(p: np.ndarray) -> np.ndarray:
    """Torna Produção um estado final (o projeto termina ao chegar nele)."""
    a = p.copy()
    a[-1] = [0, 0, 0, 1]
    return a


def semanas_esperadas(p: np.ndarray) -> np.ndarray:
    """Média de semanas até Produção, partindo de cada estado (matriz fundamental)."""
    q = absorvente(p)[:-1, :-1]
    return np.linalg.inv(np.eye(len(q)) - q).sum(axis=1)


def acumulada(p: np.ndarray, inicio: int = 0, n: int = MAX_SEMANAS) -> np.ndarray:
    """P(chegar a Produção em até k semanas), para k = 1..n."""
    a = absorvente(p)
    v = np.zeros(len(a))
    v[inicio] = 1.0
    saida = []
    for _ in range(n):
        v = v @ a
        saida.append(v[-1])
    return np.array(saida)


def simular(p: np.ndarray, n: int, semente: int = 42) -> np.ndarray:
    """Semanas até Produção em n execuções do projeto, partindo de Amostra."""
    rng = np.random.default_rng(semente)
    a = absorvente(p)
    semanas = np.zeros(n, dtype=int)
    for i in range(n):
        estado, t = 0, 0
        while estado != len(a) - 1:
            estado = rng.choice(len(a), p=a[estado])
            t += 1
        semanas[i] = t
    return semanas


def resumo(p: np.ndarray, cum: np.ndarray) -> dict:
    pmf = np.diff(np.concatenate([[0], cum]))
    k = np.arange(1, len(cum) + 1)
    return {
        "media_semanas": float((pmf * k).sum()),
        "p50": int(np.searchsorted(cum, 0.5)) + 1,
        "p80": int(np.searchsorted(cum, 0.8)) + 1,
        "p95": int(np.searchsorted(cum, 0.95)) + 1,
        "prob_ate_4": float(cum[SEMANA_MIN - 1]),
        "prob_ate_16": float(cum[SEMANA_MAX - 1]),
        "prob_passar_de_16": float(1 - cum[SEMANA_MAX - 1]),
        "prob_entre_4_e_16": float(cum[SEMANA_MAX - 1] - cum[SEMANA_MIN - 2]),
    }


def cenarios() -> pd.DataFrame:
    """Efeito de reduzir o retrabalho do Treino, que é o que o artigo propõe testar."""
    def com(tr_am, tr_tr, tr_va):
        q = P.copy()
        q[1] = [tr_am, tr_tr, tr_va, 0.0]
        return q
    casos = {
        "Original": P,
        "Menos volta do Treino à Amostra (0,2 para 0,1)": com(0.1, 0.5, 0.4),
        "Menos repetição do Treino (0,5 para 0,35)": com(0.2, 0.35, 0.45),
        "As duas melhorias": com(0.1, 0.35, 0.55),
    }
    linhas = []
    for nome, q in casos.items():
        r = resumo(q, acumulada(q))
        linhas.append({"cenario": nome, **r})
    return pd.DataFrame(linhas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulacoes", type=int, default=10000)
    ap.add_argument("--saida", default="resultados")
    args = ap.parse_args()
    saida = Path(__file__).parent / args.saida
    saida.mkdir(exist_ok=True)

    assert np.allclose(P.sum(axis=1), 1), "cada linha da matriz tem de somar 1"
    cum = acumulada(P)
    ex = resumo(P, cum)
    esperadas = semanas_esperadas(P)
    sim = simular(P, args.simulacoes)
    ex["esperadas_por_estado"] = dict(zip(ESTADOS[:-1], map(float, esperadas)))
    ex["simulacao"] = {
        "n": args.simulacoes,
        "media_semanas": float(sim.mean()),
        "maximo": int(sim.max()),
        "prob_passar_de_16": float((sim > SEMANA_MAX).mean()),
        "prob_entre_4_e_16": float(((sim >= SEMANA_MIN) & (sim <= SEMANA_MAX)).mean()),
    }
    (saida / "resumo.json").write_text(json.dumps(ex, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"semana": np.arange(1, 121), "acumulada": cum[:120],
                  "massa": np.diff(np.concatenate([[0], cum]))[:120]}).to_csv(saida / "distribuicao.csv", index=False)
    pd.DataFrame({"semanas": sim}).to_csv(saida / "simulacao.csv", index=False)
    cenarios().to_csv(saida / "cenarios.csv", index=False)

    print(f"média exata {ex['media_semanas']:.2f} semanas | simulada {sim.mean():.2f} (máximo {sim.max()})")
    print(f"passa de {SEMANA_MAX} semanas: exato {ex['prob_passar_de_16']:.1%} | simulado {ex['simulacao']['prob_passar_de_16']:.1%}")
    print(f"termina antes de {SEMANA_MIN}: {ex['prob_ate_4'] - 0:.1%} até 4 semanas | entre 4 e 16: {ex['prob_entre_4_e_16']:.1%}")
    print(f"p50 {ex['p50']} | p80 {ex['p80']} | p95 {ex['p95']} semanas")


if __name__ == "__main__":
    main()
