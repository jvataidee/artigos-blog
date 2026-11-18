"""Resume o experimento de HPO a partir de resultados/*.csv: médias, diferenças pareadas e velocidade de convergência.

Uso: python resumo.py
Grava resultados/resumo_metodos.csv e resultados/diferencas.csv e imprime as tabelas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).parent / "resultados"
REF = "Busca aleatória"


def ic(v, n: int = 4000, semente: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(semente)
    v = np.asarray(v, float)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    m = pd.read_csv(AQUI / "melhores.csv")
    conv = pd.read_csv(AQUI / "convergencia.csv")
    pad = pd.read_csv(AQUI / "padrao.csv").iloc[0]

    linhas = []
    for nome, g in m.groupby("metodo"):
        c = conv[conv.metodo == nome]
        # primeira avaliação em que o melhor valor da execução passa de 0,66 de PR-AUC na validação cruzada
        alcancou = c[c.melhor_ate_aqui >= 0.66].groupby("semente").avaliacao.min()
        linhas.append({
            "metodo": nome,
            "cv_media": g.cv_pr_auc.mean(), "cv_desvio": g.cv_pr_auc.std(),
            "teste_media": g.teste_pr_auc.mean(), "teste_desvio": g.teste_pr_auc.std(),
            "teste_min": g.teste_pr_auc.min(), "teste_max": g.teste_pr_auc.max(),
            "execucoes_que_chegam_a_066": int(len(alcancou)), "avaliacoes_mediana_ate_066": float(alcancou.median()) if len(alcancou) else float("nan"),
            "melhor_cv_com_20_avaliacoes": float(c[c.avaliacao == 20].melhor_ate_aqui.mean()),
        })
    res = pd.DataFrame(linhas).sort_values("cv_media", ascending=False)
    res.to_csv(AQUI / "resumo_metodos.csv", index=False)

    base = m[m.metodo == REF].set_index("semente")
    dif = []
    for nome, g in m.groupby("metodo"):
        if nome == REF:
            continue
        g = g.set_index("semente")
        for col in ("cv_pr_auc", "teste_pr_auc"):
            d = (g[col] - base[col]).values
            lo, hi = ic(d)
            dif.append({"metodo": nome, "medida": col, "dif_vs_busca_aleatoria": float(d.mean()), "ic_inf": lo, "ic_sup": hi, "vence": float((d > 0).mean())})
    pd.DataFrame(dif).to_csv(AQUI / "diferencas.csv", index=False)

    print(f"padrão do LightGBM: CV {pad.cv_pr_auc:.4f} | teste {pad.teste_pr_auc:.4f}")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame(dif).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    ganho = m.groupby("metodo").teste_pr_auc.mean() - pad.teste_pr_auc
    print("ganho no teste sobre o padrão:", ganho.round(4).to_dict())


if __name__ == "__main__":
    main()
