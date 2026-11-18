"""Cubo exploratório até 2024, sem consultar o bloco final de 2025."""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from protocolo import dekad, reflectance

HERE = Path(__file__).parent


def load_cube(until="2025-01-01"):
    with np.load(HERE / "dados" / "cubo_soja_s2.npz") as z:
        meta = pd.DataFrame({"data": pd.to_datetime(z["data"]), "baseline": z["baseline"],
                             "position": np.arange(len(z["data"]))})
        meta = meta[meta["data"] < pd.Timestamp(until)]
        meta = meta.sort_values(["data", "baseline"]).drop_duplicates("data", keep="last")
        pos = meta["position"].to_numpy()
        base = meta["baseline"].to_numpy()[:, None, None]
        red = reflectance(z["B04"][pos], base)
        nir = reflectance(z["B08"][pos], base)
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (nir - red) / (nir + red)
        ndvi[~np.isin(z["SCL"][pos], [4, 5])] = np.nan
    dates = pd.DatetimeIndex(meta["data"])
    grid = pd.DatetimeIndex([pd.Timestamp(y, m, d) for y in range(2019, 2026)
                             for m in range(1, 13) for d in (1, 11, 21)])
    grid = grid[grid < pd.Timestamp(until)]
    keys = np.array([dekad(d) for d in dates])
    cube = np.full((len(grid), 32, 32), np.nan)
    for i, g in enumerate(grid):
        sel = keys == g
        if sel.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                cube[i] = np.nanmax(ndvi[sel], axis=0)
    return grid, cube


def main():
    grid, cube = load_cube()
    series = pd.read_csv(HERE / "dados" / "ndvi_decendial.csv", index_col=0, parse_dates=True)["ndvi"].reindex(grid)
    center = pd.Series(cube[:, 16, 16], index=grid)
    np.testing.assert_allclose(center, series, atol=1e-12, rtol=0, equal_nan=True)
    print(f"pixel central conferido até 2024; cubo {cube.shape}")
    train_end, T = 144, len(grid)
    colors = ["#4c72b0", "#dd8452"]
    valid = np.isfinite(cube).mean(axis=(1, 2)) > 0.5
    frames = sorted(set(range(0, T, 18)) | {train_end - 1, T - 1})
    frames = [i if valid[i] else next((j for j in range(i, max(i - 6, -1), -1) if valid[j]), i)
              for i in frames]
    frames = sorted(set(frames))
    x, y = np.meshgrid(np.arange(32), np.arange(32))
    fig = plt.figure(figsize=(8, 9))
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False
    ax.set_proj_type("ortho")
    for t in frames:
        ax.contourf(x, y, cube[t], zdir="z", offset=t, levels=np.linspace(0, 1, 11),
                    cmap="YlGn", extend="both", zorder=1)
    for z, color in zip([train_end - 1 + 0.05, T - 1 + 0.05], colors):
        xx, yy = np.meshgrid([-1.5, 32.5], [-1.5, 32.5])
        ax.plot_surface(xx, yy, np.full_like(xx, z, dtype=float), color=color,
                        alpha=0.15, shade=False, zorder=2)
        ax.plot([-1.5, 32.5, 32.5, -1.5, -1.5], [-1.5, -1.5, 32.5, 32.5, -1.5],
                 [z] * 5, color=color, lw=2.5, zorder=3)
    ax.plot([16, 16], [16, 16], [0, T - 1], color="#d62728", lw=2.2, zorder=5)
    ax.scatter([16] * len(frames), [16] * len(frames), frames, color="#d62728",
                s=18, zorder=6, depthshade=False)
    ax.set_xlim(-1.5, 32.5)
    ax.set_ylim(-1.5, 32.5)
    ax.set_zlim(-0.5, T + 4)
    ax.set_box_aspect((32, 32, 58))
    ax.view_init(elev=18, azim=-40)
    ax.set_xticks([0, 16, 31])
    ax.set_yticks([0, 16, 31])
    ax.set_xlabel("x (pixels de 10 m)", fontsize=8)
    ax.set_ylabel("y (pixels de 10 m)", fontsize=8)
    ax.set_zticks([(train_end - 1) / 2, (train_end + T - 1) / 2])
    ax.set_zticklabels(["treino\n2019 a 2022", "validação\n2023 e 2024"], fontsize=8)
    for lab, color in zip(ax.get_zticklabels(), colors):
        lab.set_color(color)
    ax.tick_params(axis="z", pad=14)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    sm = plt.cm.ScalarMappable(cmap="YlGn", norm=plt.Normalize(0, 1))
    cax = fig.add_axes((0.88, 0.32, 0.02, 0.36))
    fig.colorbar(sm, cax=cax).set_label("NDVI")
    handles = [plt.Line2D([0], [0], color="#d62728", lw=2.2, marker="o", ms=5,
                           label="pixel da série (centro do cubo)"),
               plt.Rectangle((0, 0), 1, 1, facecolor=colors[0], alpha=0.5, label="fim do treino"),
               plt.Rectangle((0, 0), 1, 1, facecolor=colors[1], alpha=0.5, label="fim da validação")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.45, -0.01))
    ax.set_position([0, 0.07, 0.86, 0.92])
    (HERE / "fig").mkdir(exist_ok=True)
    fig.savefig(HERE / "fig" / "img-03.png", dpi=130, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("quadros desenhados:", [str(grid[i].date()) for i in frames])


if __name__ == "__main__":
    main()
