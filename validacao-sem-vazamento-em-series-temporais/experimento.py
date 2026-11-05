# Experimento do artigo: um pixel de soja do Sentinel-2 (MT), NDVI decendial, 2019 a 2025.
# Cada bloco "# %%" vira um trecho do post. Entrada: pixel_soja_s2.csv (baixar_pixel.py).
# %% NDVI
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

HERE = Path(__file__).parent
OUT = HERE / "fig"; OUT.mkdir(exist_ok=True)

df = pd.read_csv(HERE / "dados" / "pixel_soja_s2.csv", parse_dates=["data"])
df = df.sort_values(["data", "baseline"]).drop_duplicates("data", keep="last")  # cena reprocessada: fica a mais nova
offset = np.where(df["baseline"] >= 4.0, 1000, 0)    # a partir da baseline 04.00 o L2A soma 1000 ao DN
red = (df["B04"] - offset) / 10_000
nir = (df["B08"] - offset) / 10_000
df["ndvi"] = (nir - red) / (nir + red)
clean = df[df["SCL"].isin([4, 5])].set_index("data")["ndvi"]   # SCL 4 = vegetação, 5 = solo exposto
print(f"cenas: {len(df)} | limpas no pixel: {len(clean)}")
print("limpas por mês:", clean.groupby(clean.index.month).size().to_dict())

# %% decêndios
def dekad(ts):
    return pd.Timestamp(ts.year, ts.month, 1 if ts.day <= 10 else 11 if ts.day <= 20 else 21)


grid = pd.DatetimeIndex([pd.Timestamp(y, m, d) for y in range(2019, 2026)
                         for m in range(1, 13) for d in (1, 11, 21)])
ndvi = clean.groupby(clean.index.map(dekad)).max().reindex(grid)   # máximo limpo do decêndio
gaps = ndvi.isna().astype(int)
longest = gaps.groupby((gaps != gaps.shift()).cumsum()).sum().max()
print(f"decêndios: {len(ndvi)} | com observação: {ndvi.notna().sum()} | maior lacuna: {longest} decêndios")

TRAIN_END = ndvi.index.get_loc(pd.Timestamp("2023-01-01"))    # treino: 2019 a 2022
VAL_END = ndvi.index.get_loc(pd.Timestamp("2025-01-01"))      # validação: 2023 e 2024 | teste selado: 2025

# figura 1: observações limpas, composição decendial e os três blocos
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.scatter(clean.index, clean, s=6, color="#999999", label="observação limpa")
ax.plot(ndvi.index, ndvi, "o-", ms=3, lw=0.8, color="#2E7D32", label="máximo do decêndio")
for a, b, c, txt in [(TRAIN_END, VAL_END, "#DD8452", "validação"), (VAL_END, len(ndvi) - 1, "#4C72B0", "teste selado")]:
    ax.axvspan(ndvi.index[a], ndvi.index[b], color=c, alpha=0.10)
    ax.text(ndvi.index[a], 1.0, f" {txt}", color=c, va="top")
ax.text(ndvi.index[0], 1.0, " treino", color="#555555", va="top")
ax.set_ylim(0, 1.02); ax.set_ylabel("NDVI"); ax.margins(x=0.01)
ax.legend(loc="lower left", frameon=False, ncol=2)
fig.tight_layout(); fig.savefig(OUT / "img-01.png", dpi=130); plt.close(fig)

# figura 2: observações limpas por mês
fig, ax = plt.subplots(figsize=(8, 3.2))
cnt = clean.groupby(clean.index.month).size().reindex(range(1, 13), fill_value=0)
ax.bar(["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"], cnt, color="#2E7D32")
for i, v in enumerate(cnt):
    ax.text(i, v + 0.8, str(v), ha="center", fontsize=9)
ax.set_ylabel("cenas limpas (2019 a 2025)"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "img-02.png", dpi=130); plt.close(fig)


# %% features
LAGS = (1, 2, 3)


def fill(series):
    """Preenche lacunas: interpola entre observações e repete a última no fim."""
    return series.interpolate(limit_area="inside").ffill()


def make_features(series):
    """Linha do decêndio t: NDVI preenchido nos 3 decêndios anteriores e o decêndio do ano."""
    f = fill(series)
    X = pd.DataFrame({f"lag_{k}": f.shift(k) for k in LAGS})
    X["dekad_of_year"] = (series.index.month - 1) * 3 + (series.index.day - 1) // 10
    return X


def row_for_day(series, o, features=make_features):
    """A linha do decêndio o montada só com o passado: o valor de o fica vazio."""
    past = series.iloc[:o]
    blank = pd.Series([np.nan], index=series.index[o:o + 1])
    return features(pd.concat([past, blank])).iloc[[-1]]


def new_model():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)


# figura 3: o que cada preenchimento enxerga dentro de uma lacuna da safra
o = ndvi.index.get_loc(pd.Timestamp("2024-02-11"))
win = slice(o - 12, o + 10)
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(ndvi.index[win], fill(ndvi).iloc[win], color="#C44E52", lw=1.5, label="interpolada com a série inteira")
past_fill = fill(ndvi.iloc[:o])
ax.plot(past_fill.index[o - 12:], past_fill.iloc[o - 12:], color="#4C72B0", lw=2,
        label="preenchida só com o passado")
ax.plot([ndvi.index[o - 1], ndvi.index[o]], [past_fill.iloc[-1]] * 2, color="#4C72B0", lw=2, ls="--")
ax.scatter(ndvi.index[win], ndvi.iloc[win], color="#222222", zorder=3, s=22, label="observação limpa")
ax.axvline(ndvi.index[o], color="#555555", lw=0.8, ls=":")
ax.text(ndvi.index[o], 0.05, " origem: prever 11/02/2024", fontsize=9)
ax.set_ylim(0, 1.02); ax.set_ylabel("NDVI"); ax.legend(loc="upper left", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "img-04.png", dpi=130); plt.close(fig)


# %% MASE
def persistence(series, start, end):
    """Previsão ingênua: o último NDVI conhecido antes de cada decêndio."""
    return pd.Series([fill(series.iloc[:o]).iloc[-1] for o in range(start, end)],
                     index=series.index[start:end])


SCALE = (ndvi.iloc[1:TRAIN_END] - persistence(ndvi, 1, TRAIN_END)).abs().mean()


def mase(y_true, y_pred):
    """MAE só onde há observação real, dividido pelo MAE da persistência no treino."""
    ok = y_true.notna()
    return np.mean(np.abs(y_true[ok] - y_pred[ok])) / SCALE


val = ndvi.iloc[TRAIN_END:VAL_END]
pers_val = persistence(ndvi, TRAIN_END, VAL_END)
last_year = pd.Series([fill(ndvi.iloc[:o]).iloc[o - 36] for o in range(TRAIN_END, VAL_END)], index=val.index)
print(f"escala: {SCALE:.3f} de NDVI | decêndios avaliados na validação: {val.notna().sum()}")
print(f"persistência: MASE = {mase(val, pers_val):.3f}")
print(f"mesmo decêndio do ano anterior: MASE = {mase(val, last_year):.3f}")


# %% KFold aleatório
X_all = make_features(ndvi)                            # lacunas preenchidas na série inteira
dev_rows = [i for i in range(len(LAGS), VAL_END) if pd.notna(ndvi.iloc[i])]
X_dev, y_dev = X_all.iloc[dev_rows], ndvi.iloc[dev_rows]
pred_kf = pd.Series(index=y_dev.index, dtype=float)
for tr, te in KFold(n_splits=5, shuffle=True, random_state=42).split(X_dev):
    pred_kf.iloc[te] = new_model().fit(X_dev.iloc[tr], y_dev.iloc[tr]).predict(X_dev.iloc[te])
pred_kf = pred_kf.reindex(val.index)
print(f"KFold aleatório: MASE = {mase(val, pred_kf):.3f}")


# %% walk-forward com as lacunas preenchidas antes
train_rows = [i for i in range(len(LAGS), TRAIN_END) if pd.notna(ndvi.iloc[i])]
model = new_model().fit(X_all.iloc[train_rows], ndvi.iloc[train_rows])
pred_pre = pd.Series(model.predict(X_all.iloc[TRAIN_END:VAL_END]), index=val.index)
print(f"walk-forward, lacunas preenchidas antes: MASE = {mase(val, pred_pre):.3f}")


# %% walk-forward só com o passado
def walk_forward(series, start, end, features=make_features, past_only_train=True):
    """Ajusta uma vez antes de start e prevê cada decêndio de [start, end) só com o passado."""
    rows = [i for i in range(len(LAGS) + 1, start) if pd.notna(series.iloc[i])]
    if past_only_train:    # cada linha de treino montada como seria no dia dela
        X_tr = pd.concat([row_for_day(series, i, features) for i in rows])
    else:                  # linhas de treino da série de treino preenchida de uma vez
        X_tr = features(series.iloc[:start]).iloc[rows]
    model = new_model().fit(X_tr, series.iloc[rows])
    preds = [model.predict(row_for_day(series, o, features))[0] for o in range(start, end)]
    return pd.Series(preds, index=series.index[start:end])


pred_mix = walk_forward(ndvi, TRAIN_END, VAL_END, past_only_train=False)
pred_wf = walk_forward(ndvi, TRAIN_END, VAL_END)
print(f"walk-forward, só a previsão com o passado: MASE = {mase(val, pred_mix):.3f}")
print(f"walk-forward, treino e previsão com o passado: MASE = {mase(val, pred_wf):.3f}")


# %% idade da última observação
def make_features_age(series):
    """As mesmas features e há quantos decêndios saiu a última observação limpa."""
    X = make_features(series)
    pos = np.arange(len(series))
    last_seen = pd.Series(np.where(series.notna(), pos, np.nan), index=series.index).ffill()
    X["obs_age"] = pos - last_seen.shift(1).to_numpy()
    return X


pred_age = walk_forward(ndvi, TRAIN_END, VAL_END, features=make_features_age)
print(f"walk-forward com idade da observação: MASE = {mase(val, pred_age):.3f}")

# figura 4: comparativo
rows = [
    ("persistência", mase(val, pers_val), "#999999"),
    ("RF, KFold aleatório", mase(val, pred_kf), "#C44E52"),
    ("RF, lacunas preenchidas antes", mase(val, pred_pre), "#C44E52"),
    ("RF, só a previsão com o passado", mase(val, pred_mix), "#C44E52"),
    ("RF, treino e previsão com o passado", mase(val, pred_wf), "#4C72B0"),
    ("RF + idade da observação", mase(val, pred_age), "#4C72B0"),
]
fig, ax = plt.subplots(figsize=(9, 3.8))
names, vals, cols = zip(*rows[::-1])
ax.barh(names, vals, color=cols)
ax.axvline(rows[0][1], color="#222222", lw=0.8, ls=":")
for i, v in enumerate(vals):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0, max(vals) * 1.12)
ax.set_xlabel("MASE na validação (2023 e 2024; menor é melhor)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "img-05.png", dpi=130); plt.close(fig)

# figura 5: previsões na validação
fig, ax = plt.subplots(figsize=(11, 3.8))
ok = val.notna()
ax.scatter(val.index[ok], val[ok], color="#222222", s=18, zorder=3, label="observado")
ax.plot(val.index, pred_pre, color="#C44E52", lw=1.2, label="RF, lacunas preenchidas antes")
ax.plot(val.index, pred_wf, color="#4C72B0", lw=1.2, label="RF, só com o passado")
ax.plot(val.index, pers_val, color="#999999", lw=1, ls="--", label="persistência")
ax.set_ylim(0, 1.02); ax.set_ylabel("NDVI"); ax.legend(loc="lower left", frameon=False, ncol=2)
ax.margins(x=0.01)
fig.tight_layout(); fig.savefig(OUT / "img-06.png", dpi=130); plt.close(fig)


# %% teste selado
sealed = ndvi.iloc[VAL_END:]
pers_test = persistence(ndvi, VAL_END, len(ndvi))
print(f"decêndios avaliados no teste: {sealed.notna().sum()}")
print(f"teste selado, persistência: MASE = {mase(sealed, pers_test):.3f}")

# %% a tentação
pred_test_age = walk_forward(ndvi, VAL_END, len(ndvi), features=make_features_age)
print(f"teste selado, RF + idade da observação: MASE = {mase(sealed, pred_test_age):.3f}")
