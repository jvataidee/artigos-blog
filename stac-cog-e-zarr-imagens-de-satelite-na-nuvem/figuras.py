"""Figuras do artigo a partir dos arquivos locais (saida/ e resultados/), sem acesso à rede.

- fig/ndvi_mensal_recife.png: NDVI mensal, com os meses sem data limpa marcados;
A capa fica em capa.py.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

PASTA = Path(__file__).parent
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def painel_mensal(ndvi, cenas_por_mes):
    fig, eixos = plt.subplots(3, 4, figsize=(12, 11.5), constrained_layout=True)
    for m, ax in enumerate(eixos.flat, start=1):
        ax.set_xticks([])
        ax.set_yticks([])
        sel = ndvi.where(ndvi.time.dt.month == m, drop=True)
        n = cenas_por_mes.get(MESES[m - 1], 0)
        if sel.sizes["time"] == 0 or np.isnan(sel.values).all():
            ax.set_facecolor("#eeeeee")
            ax.text(0.5, 0.5, "sem data com\nRecife limpa", ha="center", va="center", transform=ax.transAxes,
                    color="#666", fontsize=10)
            ax.set_title(f"{MESES[m - 1]}/2025", fontsize=11)
            continue
        im = ax.imshow(sel.isel(time=0).values, cmap="RdYlGn", vmin=0, vmax=0.9, interpolation="nearest")
        ax.set_title(f"{MESES[m - 1]}/2025 ({n} data{'s' if n > 1 else ''})", fontsize=11)
    fig.colorbar(im, ax=eixos, shrink=0.5, label="NDVI (mediana do mês)")
    fig.savefig(PASTA / "fig" / "ndvi_mensal_recife.png", dpi=100)


def main():
    (PASTA / "fig").mkdir(exist_ok=True)
    ndvi = xr.open_zarr(PASTA / "saida" / "ndvi_recife_2025.zarr")["ndvi"].load()
    cubo = json.loads((PASTA / "resultados" / "cubo.json").read_text(encoding="utf-8"))
    cenas_por_mes = {m["mes"]: m["datas"] for m in cubo["por_mes"]}
    painel_mensal(ndvi, cenas_por_mes)


if __name__ == "__main__":
    main()
