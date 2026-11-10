"""Passo 3: triagem de nuvens na área de Recife, lendo a SCL em baixa resolução.

O eo:cloud_cover do item vale para a quadrícula inteira (cerca de 110 x 110 km). Aqui
lemos a camada SCL só na área de Recife e a 200 m, o que faz o leitor usar as overviews
dos COGs. Grava resultados/triagem.csv, resultados/triagem.json e a figura em fig/.
"""

import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import planetary_computer
import pystac

PASTA = Path(__file__).parent
BBOX = [-35.02, -8.15, -34.86, -7.95]
LIMIAR = 0.5  # fração mínima da área de Recife com pixel limpo


def main():
    itens = pystac.ItemCollection.from_file(str(PASTA / "dados" / "itens_recife_2025.json"))
    itens = [planetary_computer.sign(i) for i in itens]

    t0 = time.perf_counter()
    scl = odc.stac.load(itens, bands=["SCL"], bbox=BBOX, resolution=200, groupby="solar_day",
                        chunks={}).SCL.compute()
    seg = time.perf_counter() - t0
    print(f"SCL de {scl.sizes['time']} datas a 200 m ({scl.sizes['x']} x {scl.sizes['y']} px) em {seg:.0f} s")

    com_dado = scl > 0  # 0 = sem dado
    limpo = scl.isin([4, 5, 6])  # vegetação, solo exposto, água
    frac = (limpo.sum(dim=["x", "y"]) / com_dado.sum(dim=["x", "y"])).values

    # eo:cloud_cover por data: o menor entre as quadrículas do dia (a data passa no filtro se um item passar)
    nuvem_cena = {}
    for i in itens:
        nuvem_cena.setdefault(i.datetime.date().isoformat(), []).append(i.properties["eo:cloud_cover"])
    datas = [str(t)[:10] for t in scl.time.values]
    cena = np.array([min(nuvem_cena[d]) for d in datas])

    linhas = [{"data": d, "nuvem_cena_pct": round(float(c), 1), "limpo_recife_pct": round(100 * float(f), 1)}
              for d, c, f in zip(datas, cena, frac)]
    with open(PASTA / "resultados" / "triagem.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)

    selecionadas = [d for d, f in zip(datas, frac) if f >= LIMIAR]
    passa_20 = [d for d, c in zip(datas, cena) if c < 20]
    print(f"datas com >= {LIMIAR:.0%} de Recife limpa: {len(selecionadas)} de {len(datas)}")
    print(f"datas com eo:cloud_cover < 20%: {len(passa_20)}")
    perdidas = [d for d in selecionadas if d not in passa_20]
    print(f"datas boas em Recife que o filtro de 20% descartaria: {len(perdidas)}")
    print("correlação (cena x Recife):", round(float(np.corrcoef(cena, 1 - frac)[0, 1]), 2))

    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    ax.scatter(cena, 100 * frac, s=28, color="#2a5bd7", alpha=0.75)
    ax.axvline(20, color="#d9532c", ls="--", lw=1.2)
    ax.axhline(100 * LIMIAR, color="#555", ls=":", lw=1.2)
    ax.text(21, 3, "eo:cloud_cover < 20%", color="#d9532c", fontsize=9)
    ax.text(62, 100 * LIMIAR + 2, f"{LIMIAR:.0%} de Recife limpa", color="#555", fontsize=9)
    ax.set_xlabel("nuvens na quadrícula inteira, eo:cloud_cover (%, menor do dia)")
    ax.set_ylabel("área de Recife com pixel limpo pela SCL (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title(f"Sentinel-2 L2A sobre Recife em 2025 ({len(datas)} datas)")
    (PASTA / "fig").mkdir(exist_ok=True)
    fig.savefig(PASTA / "fig" / "triagem_nuvens.png", dpi=130)

    resumo = {"datas": len(datas), "grade_px": [int(scl.sizes["x"]), int(scl.sizes["y"])],
              "segundos": round(seg, 1), "limiar": LIMIAR, "selecionadas": selecionadas,
              "datas_lt20": passa_20, "boas_descartadas_por_lt20": len(perdidas)}
    (PASTA / "resultados" / "triagem.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")


if __name__ == "__main__":
    main()
