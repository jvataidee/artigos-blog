"""Análises da auditoria: KFold só com o passado, sementes, bootstrap por blocos,
erro por fase da cultura e coerência do talhão no cubo."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
src = (HERE / "experimento.py").read_text(encoding="utf-8")
src = src.split("# %% teste selado")[0]
# não refaz figuras nem prints do experimento
import matplotlib
matplotlib.use("Agg")
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, "experimento.py", "exec"))

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

sealed = ndvi.iloc[VAL_END:]
pers_test = persistence(ndvi, VAL_END, len(ndvi))

# %% 1. KFold aleatório com linhas montadas só com o passado
dev_rows_p = [i for i in range(len(LAGS) + 1, VAL_END) if pd.notna(ndvi.iloc[i])]
X_dev_p = pd.concat([row_for_day(ndvi, i) for i in dev_rows_p])
y_dev_p = ndvi.iloc[dev_rows_p]
pred_kf_p = pd.Series(index=y_dev_p.index, dtype=float)
for tr, te in KFold(n_splits=5, shuffle=True, random_state=42).split(X_dev_p):
    pred_kf_p.iloc[te] = new_model().fit(X_dev_p.iloc[tr], y_dev_p.iloc[tr]).predict(X_dev_p.iloc[te])
pred_kf_p = pred_kf_p.reindex(val.index)
print(f"KFold aleatório, linhas só com o passado: MASE = {mase(val, pred_kf_p):.3f}")


# %% 2a. sementes do Random Forest
def run_all(seed):
    global new_model
    def new_model():
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    out = {}
    pk = pd.Series(index=y_dev.index, dtype=float)
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=seed).split(X_dev):
        pk.iloc[te] = new_model().fit(X_dev.iloc[tr], y_dev.iloc[tr]).predict(X_dev.iloc[te])
    out["kfold"] = mase(val, pk.reindex(val.index))
    pkp = pd.Series(index=y_dev_p.index, dtype=float)
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=seed).split(X_dev_p):
        pkp.iloc[te] = new_model().fit(X_dev_p.iloc[tr], y_dev_p.iloc[tr]).predict(X_dev_p.iloc[te])
    out["kfold_passado"] = mase(val, pkp.reindex(val.index))
    m = new_model().fit(X_all.iloc[train_rows], ndvi.iloc[train_rows])
    out["pre"] = mase(val, pd.Series(m.predict(X_all.iloc[TRAIN_END:VAL_END]), index=val.index))
    out["mix"] = mase(val, walk_forward(ndvi, TRAIN_END, VAL_END, past_only_train=False))
    out["wf"] = mase(val, walk_forward(ndvi, TRAIN_END, VAL_END))
    out["idade"] = mase(val, walk_forward(ndvi, TRAIN_END, VAL_END, features=make_features_age))
    out["idade_teste"] = mase(sealed, walk_forward(ndvi, VAL_END, len(ndvi), features=make_features_age))
    return out


seeds = pd.DataFrame([run_all(s) for s in range(10)])
print("\n10 sementes (mín | mediana | máx):")
print(seeds.agg(["min", "median", "max"]).T.round(3).to_string())
print(f"persistência: validação {mase(val, pers_val):.3f} | teste {mase(sealed, pers_test):.3f}")

# restaura a semente original
def new_model():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)


# %% 2b. bootstrap por blocos da diferença de MASE (modelo - persistência)
def block_bootstrap_diff(y_true, pred, ref, block=6, n=5000, seed=0):
    """Reamostra blocos de `block` decêndios consecutivos; devolve IC 95% de MASE(pred) - MASE(ref)."""
    rng = np.random.default_rng(seed)
    ok = y_true.notna().to_numpy()
    e_m = np.abs(y_true - pred).to_numpy() / SCALE
    e_r = np.abs(y_true - ref).to_numpy() / SCALE
    T = len(y_true)
    starts = np.arange(T - block + 1)
    diffs = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, size=int(np.ceil(T / block)))])[:T]
        sel = idx[ok[idx]]
        if len(sel) == 0:
            continue
        diffs.append(e_m[sel].mean() - e_r[sel].mean())
    diffs = np.array(diffs)
    point = e_m[ok].mean() - e_r[ok].mean()
    return point, np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), (diffs < 0).mean()


pred_test_age = walk_forward(ndvi, VAL_END, len(ndvi), features=make_features_age)
print("\nbootstrap por blocos de 6 decêndios, diferença de MASE em relação à persistência:")
for name, y_t, p, r in [
    ("KFold aleatório (validação)", val, pred_kf, pers_val),
    ("KFold só com o passado (validação)", val, pred_kf_p, pers_val),
    ("lacunas preenchidas antes (validação)", val, pred_pre, pers_val),
    ("treino e previsão com o passado (validação)", val, pred_wf, pers_val),
    ("RF + idade (validação)", val, pred_age, pers_val),
    ("RF + idade (teste)", sealed, pred_test_age, pers_test),
]:
    d, lo, hi, p_neg = block_bootstrap_diff(y_t, p, r)
    print(f"  {name}: {d:+.3f}  IC95% [{lo:+.3f}; {hi:+.3f}]  P(modelo melhor) = {p_neg:.2f}")


# %% 4. erro por fase da cultura
def phase(idx):
    return np.where(idx.month.isin([10, 11, 12, 1, 2]), "safra de soja (out a fev)", "resto do ano (mar a set)")


print("\nMASE por fase na validação:")
rows = []
for name, p in [("persistência", pers_val), ("lacunas preenchidas antes", pred_pre),
                ("treino e previsão com o passado", pred_wf), ("RF + idade da observação", pred_age)]:
    for ph in ["safra de soja (out a fev)", "resto do ano (mar a set)"]:
        m = phase(val.index) == ph
        rows.append((name, ph, mase(val[m], p[m]), int(val[m].notna().sum())))
tab = pd.DataFrame(rows, columns=["modelo", "fase", "MASE", "n"]).pivot(index="modelo", columns="fase", values="MASE")
print(tab.round(3).to_string())
print("pontos observados por fase:", {ph: int(val[phase(val.index) == ph].notna().sum())
                                       for ph in ["safra de soja (out a fev)", "resto do ano (mar a set)"]})

# %% 5. coerência do talhão no cubo
exec((HERE / "figura_cubo.py").read_text(encoding="utf-8").split("# conferência")[0])
center = cube[:, 16, 16]
corr = np.full((32, 32), np.nan)
for r_ in range(32):
    for c_ in range(32):
        s = cube[:, r_, c_]
        ok = np.isfinite(s) & np.isfinite(center)
        if ok.sum() > 30:
            corr[r_, c_] = np.corrcoef(s[ok], center[ok])[0, 1]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print(f"\ncubo: correlação dos 1.024 pixels com o pixel central: mediana {np.nanmedian(corr):.2f}, "
          f"fração com r > 0,9: {np.nanmean(corr > 0.9):.2f}, fração com r > 0,8: {np.nanmean(corr > 0.8):.2f}")
