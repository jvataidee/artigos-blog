"""Cubo NDVI decendial 32x32 com os blocos treino/validação/teste e o pixel da série marcado."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
z = np.load(HERE / "dados" / "cubo_soja_s2.npz")
dates = pd.to_datetime(z["data"])
keep = ~pd.Series(dates).duplicated(keep="last").to_numpy()      # cena reprocessada: fica a mais nova
dates, base = dates[keep], z["baseline"][keep]
b4, b8, scl = (z[k][keep].astype(float) for k in ("B04", "B08", "SCL"))
off = np.where(base >= 4.0, 1000, 0)[:, None, None]
red, nir = (b4 - off) / 1e4, (b8 - off) / 1e4
ndvi = (nir - red) / (nir + red)
ndvi[~np.isin(scl, [4, 5])] = np.nan


def dekad(ts):
    return pd.Timestamp(ts.year, ts.month, 1 if ts.day <= 10 else 11 if ts.day <= 20 else 21)


grid = pd.DatetimeIndex([pd.Timestamp(y, m, d) for y in range(2019, 2026)
                         for m in range(1, 13) for d in (1, 11, 21)])
keys = np.array([dekad(d) for d in dates])
cube = np.full((len(grid), 32, 32), np.nan)
for i, g in enumerate(grid):
    sel = keys == g
    if sel.any():
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cube[i] = np.nanmax(ndvi[sel], axis=0)

# conferência: o pixel central é a série do artigo
serie = pd.read_csv(HERE / "dados" / "ndvi_decendial.csv", index_col=0, parse_dates=True)["ndvi"]
centro = pd.Series(cube[:, 16, 16], index=grid)
diff = (centro - serie).abs().max()
print(f"pixel central x série do artigo: diferença máxima {diff:.2e}, "
      f"mesmas lacunas: {(centro.isna() == serie.isna()).all()}")
print(f"cubo: {cube.shape} | decêndios com algum pixel limpo: {np.isfinite(cube).any(axis=(1, 2)).sum()}")

TRAIN_END, VAL_END, T = 144, 216, len(grid)
COL = {"train": "#4c72b0", "eval": "#dd8452", "test": "#c44e52"}

# quadros: um a cada 12 decêndios com algum dado, mais os limites dos blocos
valid = np.isfinite(cube).mean(axis=(1, 2)) > 0.5
frames = sorted({i for i in range(0, T, 18)} | {TRAIN_END - 1, VAL_END - 1, T - 1})
frames = [i if valid[i] else next((j for j in range(i, max(i - 6, -1), -1) if valid[j]), i) for i in frames]
frames = sorted(set(frames))

x, y = np.meshgrid(np.arange(32), np.arange(32))
fig = plt.figure(figsize=(8, 9))
ax = fig.add_subplot(projection="3d")
ax.computed_zorder = False
ax.set_proj_type("ortho")
levels = np.linspace(0, 1, 11)
for t in frames:
    ax.contourf(x, y, cube[t], zdir="z", offset=t, levels=levels, cmap="YlGn", extend="both", zorder=1)


def plane(zv, color):
    xx, yy = np.meshgrid([-1.5, 32.5], [-1.5, 32.5])
    ax.plot_surface(xx, yy, np.full_like(xx, zv, dtype=float), color=color, alpha=0.15, shade=False, zorder=2)
    ax.plot([-1.5, 32.5, 32.5, -1.5, -1.5], [-1.5, -1.5, 32.5, 32.5, -1.5], [zv] * 5, color=color, lw=2.5, zorder=3)


plane(TRAIN_END - 1 + 0.05, COL["train"])
plane(VAL_END - 1 + 0.05, COL["eval"])
plane(T - 1 + 0.05, COL["test"])

# o pixel da série atravessando o tempo
ax.plot([16, 16], [16, 16], [0, T - 1], color="#d62728", lw=2.2, zorder=5)
ax.scatter([16] * len(frames), [16] * len(frames), frames, color="#d62728", s=18, zorder=6, depthshade=False)

ax.set_xlim(-1.5, 32.5); ax.set_ylim(-1.5, 32.5); ax.set_zlim(-0.5, T + 4)
ax.set_box_aspect((32, 32, 58))
ax.view_init(elev=18, azim=-40)
ax.set_xticks([0, 16, 31]); ax.set_yticks([0, 16, 31])
ax.set_xlabel("x (pixels de 10 m)", fontsize=8); ax.set_ylabel("y (pixels de 10 m)", fontsize=8)
zt = [(TRAIN_END - 1) / 2, (TRAIN_END + VAL_END - 1) / 2, (VAL_END + T - 1) / 2]
ax.set_zticks(zt)
ax.set_zticklabels(["treino\n2019 a 2022", "validação\n2023 e 2024", "teste selado\n2025"], fontsize=8)
for lab, c in zip(ax.get_zticklabels(), COL.values()):
    lab.set_color(c)
ax.tick_params(axis="z", pad=14)
ax.tick_params(axis="x", labelsize=7); ax.tick_params(axis="y", labelsize=7)

sm = plt.cm.ScalarMappable(cmap="YlGn", norm=plt.Normalize(0, 1))
cax = fig.add_axes((0.88, 0.32, 0.02, 0.36))
fig.colorbar(sm, cax=cax).set_label("NDVI")
handles = [
    plt.Line2D([0], [0], color="#d62728", lw=2.2, marker="o", ms=5, label="pixel da série (centro do cubo)"),
    plt.Rectangle((0, 0), 1, 1, facecolor=COL["train"], alpha=0.5, label="fim do treino"),
    plt.Rectangle((0, 0), 1, 1, facecolor=COL["eval"], alpha=0.5, label="fim da validação"),
    plt.Rectangle((0, 0), 1, 1, facecolor=COL["test"], alpha=0.5, label="fim do teste selado"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.45, -0.01))
ax.set_position([0.0, 0.07, 0.86, 0.92])
(HERE / "fig").mkdir(exist_ok=True)
fig.savefig(HERE / "fig" / "img-03.png", dpi=130, bbox_inches="tight", pad_inches=0.15)
print("quadros desenhados:", [str(grid[i].date()) for i in frames])
