"""Experimento do artigo. Desenvolvimento até 2024; avaliação final em 2025."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from protocolo import (
    fill,
    load_series,
    make_features_age,
    mase,
    persistence,
    run_development,
    walk_forward,
)

HERE = Path(__file__).parent


def main():
    out = HERE / "fig"
    out.mkdir(exist_ok=True)
    df, clean, ndvi = load_series(HERE / "dados" / "pixel_soja_s2.csv")
    train_end = ndvi.index.get_loc(pd.Timestamp("2023-01-01"))
    val_end = ndvi.index.get_loc(pd.Timestamp("2025-01-01"))
    # Nem as figuras exploratórias nem os protocolos deliberadamente incorretos recebem 2025.
    dev = ndvi.iloc[:val_end]
    clean_dev = clean[clean.index < pd.Timestamp("2025-01-01")]
    df_dev = df[df["data"] < pd.Timestamp("2025-01-01")]
    gaps = dev.isna().astype(int)
    longest = gaps.groupby((gaps != gaps.shift()).cumsum()).sum().max()
    print(f"até 2024: cenas {len(df_dev)} | limpas no pixel {len(clean_dev)}")
    print("limpas por mês até 2024:", clean_dev.groupby(clean_dev.index.month).size().to_dict())
    print(f"desenvolvimento: decêndios {len(dev)} | observados {dev.notna().sum()} | maior lacuna {longest}")

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.scatter(clean_dev.index, clean_dev, s=6, color="#999999", label="observação limpa")
    ax.plot(dev.index, dev, "o-", ms=3, lw=0.8, color="#2E7D32", label="máximo do decêndio")
    ax.axvspan(dev.index[train_end], pd.Timestamp("2025-01-01"), color="#DD8452", alpha=0.10)
    ax.text(dev.index[train_end], 1, " validação (2023 e 2024)", color="#DD8452", va="top")
    ax.text(dev.index[0], 1, " treino (2019 a 2022)", color="#555555", va="top")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("NDVI")
    ax.margins(x=0.01)
    ax.legend(loc="lower left", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "img-01.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    cnt = clean_dev.groupby(clean_dev.index.month).size().reindex(range(1, 13), fill_value=0)
    ax.bar(["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"], cnt, color="#2E7D32")
    for i, v in enumerate(cnt):
        ax.text(i, v + 0.8, str(v), ha="center", fontsize=9)
    ax.set_ylabel("cenas limpas (2019 a 2024)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "img-02.png", dpi=130)
    plt.close(fig)

    result = run_development(dev, train_end, val_end)
    val, scale, p = result["val"], result["scale"], result["predictions"]
    print(f"escala: {scale:.3f} de NDVI | decêndios avaliados na validação: {val.notna().sum()}")
    names = {"persistencia": "persistência", "ano_anterior": "mesmo decêndio do ano anterior",
             "kfold": "KFold aleatório", "kfold_passado": "KFold aleatório, linhas só com o passado",
             "pre": "walk-forward, lacunas preenchidas antes", "mix": "walk-forward, só a previsão com o passado",
             "wf": "walk-forward, treino e previsão com o passado", "idade": "walk-forward com idade da observação"}
    for key, name in names.items():
        print(f"{name}: MASE = {mase(val, p[key], scale):.3f}")

    o = dev.index.get_loc(pd.Timestamp("2024-02-11"))
    win = slice(o - 12, o + 10)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(dev.index[win], fill(dev).iloc[win], color="#C44E52", lw=1.5,
            label="interpolada com todo o desenvolvimento")
    past_fill = fill(dev.iloc[:o])
    ax.plot(past_fill.index[o - 12:], past_fill.iloc[o - 12:], color="#4C72B0", lw=2,
            label="preenchida só com o passado")
    ax.plot([dev.index[o - 1], dev.index[o]], [past_fill.iloc[-1]] * 2, color="#4C72B0", lw=2, ls="--")
    ax.scatter(dev.index[win], dev.iloc[win], color="#222222", zorder=3, s=22, label="observação limpa")
    ax.axvline(dev.index[o], color="#555555", lw=0.8, ls=":")
    ax.text(dev.index[o], 0.05, " origem: prever o decêndio de 11/02/2024", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("NDVI")
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "img-04.png", dpi=130)
    plt.close(fig)

    rows = [("persistência", "persistencia", "#999999"),
            ("RF, KFold aleatório", "kfold", "#C44E52"),
            ("RF, KFold com entradas causais", "kfold_passado", "#C44E52"),
            ("RF, lacunas preenchidas antes", "pre", "#C44E52"),
            ("RF, só a previsão com o passado", "mix", "#C44E52"),
            ("RF, treino e previsão com o passado", "wf", "#4C72B0"),
            ("RF + idade da observação", "idade", "#4C72B0")]
    fig, ax = plt.subplots(figsize=(10, 4.3))
    values = [mase(val, p[key], scale) for _, key, _ in rows[::-1]]
    ax.barh([name for name, _, _ in rows[::-1]], values, color=[c for _, _, c in rows[::-1]])
    ax.axvline(mase(val, p["persistencia"], scale), color="#222222", lw=0.8, ls=":")
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, max(values) * 1.12)
    ax.set_xlabel("MASE na validação (2023 e 2024; menor é melhor)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "img-05.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ok = val.notna()
    ax.scatter(val.index[ok], val[ok], color="#222222", s=18, zorder=3, label="observado")
    for key, color, label, style in [("pre", "#C44E52", "RF, lacunas preenchidas antes", "-"),
                                     ("wf", "#4C72B0", "RF, só com o passado", "-"),
                                     ("persistencia", "#999999", "persistência", "--")]:
        ax.plot(val.index, p[key], color=color, lw=1.2, label=label, ls=style)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("NDVI")
    ax.legend(loc="lower left", frameon=False, ncol=2)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out / "img-06.png", dpi=130)
    plt.close(fig)

    # Decisão tomada somente com a validação. A comparação adicional no teste é descritiva.
    selected = min(("persistencia", "wf", "idade"), key=lambda key: mase(val, p[key], scale))
    print("seleção pela validação:", selected)
    sealed = ndvi.iloc[val_end:]
    pers_test = persistence(ndvi, val_end, len(ndvi))
    pred_test_age = walk_forward(ndvi, val_end, len(ndvi), features=make_features_age)
    print(f"decêndios avaliados no teste: {sealed.notna().sum()}")
    print(f"teste selado, persistência: MASE = {mase(sealed, pers_test, scale):.3f}")
    print(f"teste selado, RF + idade da observação: MASE = {mase(sealed, pred_test_age, scale):.3f}")
    print("Observação: a execução reproduz uma análise já publicada; não cria um novo teste prospectivo.")


if __name__ == "__main__":
    main()
