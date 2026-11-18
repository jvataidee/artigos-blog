"""Funções do experimento, sem leituras, ajuste ou avaliação na importação."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

LAGS = (1, 2, 3)
SEEDS = tuple(range(10))


def reflectance(dn, baseline):
    """Política do arquivo L2A: DN/10000 antes de 04.00; (DN-1000)/10000 depois.

    A versão de processamento não substitui os metadados radiométricos do produto.
    Esta política deve ser conferida ao trocar a coleção ou os arquivos de entrada.
    """
    baseline = np.asarray(baseline, dtype=float)
    if not np.isfinite(baseline).all():
        raise ValueError("Baseline ausente ou inválida; não é seguro inferir o offset.")
    return (np.asarray(dn, dtype=float) - np.where(baseline >= 4.0, 1000, 0)) / 10000


def dekad(ts):
    return pd.Timestamp(ts.year, ts.month, 1 if ts.day <= 10 else 11 if ts.day <= 20 else 21)


def load_series(path):
    df = pd.read_csv(Path(path), parse_dates=["data"])
    df["baseline"] = pd.to_numeric(df["baseline"], errors="raise")
    df = df.sort_values(["data", "baseline"]).drop_duplicates("data", keep="last")
    red = reflectance(df["B04"], df["baseline"])
    nir = reflectance(df["B08"], df["baseline"])
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ndvi"] = (nir - red) / (nir + red)
    clean = df[df["SCL"].isin([4, 5])].set_index("data")["ndvi"]
    grid = pd.DatetimeIndex([pd.Timestamp(y, m, d) for y in range(2019, 2026)
                             for m in range(1, 13) for d in (1, 11, 21)])
    return df, clean, clean.groupby(clean.index.map(dekad)).max().reindex(grid)


def fill(series):
    """Interpola lacunas internas e repete a última observação no fim."""
    return series.interpolate(limit_area="inside").ffill()


def make_features(series):
    f = fill(series)
    X = pd.DataFrame({f"lag_{k}": f.shift(k) for k in LAGS})
    X["dekad_of_year"] = (series.index.month - 1) * 3 + (series.index.day - 1) // 10
    return X


def row_for_day(series, o, features=make_features):
    """Monta a entrada de o sem consultar seu alvo nem os instantes posteriores."""
    past = series.iloc[:o]
    blank = pd.Series([np.nan], index=series.index[o:o + 1])
    return features(pd.concat([past, blank])).iloc[[-1]]


def make_features_age(series):
    X = make_features(series)
    pos = np.arange(len(series))
    last_seen = pd.Series(np.where(series.notna(), pos, np.nan), index=series.index).ffill()
    X["obs_age"] = pos - last_seen.shift(1).to_numpy()
    return X


def eligible_rows(series, end):
    """Mesmo conjunto de alvos em todas as condições experimentais."""
    return [i for i in range(max(LAGS) + 1, end) if pd.notna(series.iloc[i])]


def new_model(seed=42, n_estimators=300):
    return RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=3,
                                 random_state=seed, n_jobs=-1)


def persistence(series, start, end):
    return pd.Series([fill(series.iloc[:o]).iloc[-1] for o in range(start, end)],
                     index=series.index[start:end])


def walk_forward(series, start, end, features=make_features, past_only_train=True,
                 seed=42, n_estimators=300):
    """Ajusta uma vez; cada entrada usa apenas dados anteriores à sua origem.

    As entradas causais são montadas separadamente. A previsão em lote é equivalente
    às chamadas por origem porque o modelo permanece fixo durante o período.
    """
    rows = eligible_rows(series, start)
    if past_only_train:
        X_tr = pd.concat([row_for_day(series, i, features) for i in rows])
    else:
        X_tr = features(series.iloc[:start]).iloc[rows]
    model = new_model(seed, n_estimators).fit(X_tr, series.iloc[rows])
    X_pred = pd.concat([row_for_day(series, o, features) for o in range(start, end)])
    return pd.Series(model.predict(X_pred), index=series.index[start:end])


def observed_errors(y_true, y_pred):
    if not y_true.index.equals(y_pred.index):
        raise ValueError("Alvos e previsões devem ter os mesmos índices.")
    ok = y_true.notna()
    if not ok.any() or not np.isfinite(y_pred[ok]).all():
        raise ValueError("É necessária uma previsão finita para cada alvo observado.")
    return (y_true[ok] - y_pred[ok]).abs()


def mase(y_true, y_pred, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("A escala do treinamento deve ser finita e positiva.")
    return observed_errors(y_true, y_pred).mean() / scale


def run_development(series, train_end, val_end, seed=42, n_estimators=300):
    """Isola 2025 antes de construir qualquer entrada ou ajustar os modelos."""
    dev = series.iloc[:val_end].copy()
    val = dev.iloc[train_end:]
    scale = (dev.iloc[1:train_end] - persistence(dev, 1, train_end)).abs().mean()
    dev_rows = eligible_rows(dev, val_end)
    train_rows = eligible_rows(dev, train_end)
    X_all = make_features(dev)
    X_dev = X_all.iloc[dev_rows]
    X_dev_p = pd.concat([row_for_day(dev, i) for i in dev_rows])
    y_dev = dev.iloc[dev_rows]
    folds = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(X_dev))
    predictions = {}
    for name, X in [("kfold", X_dev), ("kfold_passado", X_dev_p)]:
        pred = pd.Series(index=y_dev.index, dtype=float)
        for tr, te in folds:  # mesmas linhas e dobras nas duas condições
            model = new_model(seed, n_estimators).fit(X.iloc[tr], y_dev.iloc[tr])
            pred.iloc[te] = model.predict(X.iloc[te])
        predictions[name] = pred.reindex(val.index)
    model = new_model(seed, n_estimators).fit(X_all.iloc[train_rows], dev.iloc[train_rows])
    predictions["pre"] = pd.Series(model.predict(X_all.iloc[train_end:]), index=val.index)
    kwargs = dict(seed=seed, n_estimators=n_estimators)
    predictions["mix"] = walk_forward(dev, train_end, val_end, past_only_train=False, **kwargs)
    predictions["wf"] = walk_forward(dev, train_end, val_end, **kwargs)
    predictions["idade"] = walk_forward(dev, train_end, val_end, features=make_features_age, **kwargs)
    predictions["persistencia"] = persistence(dev, train_end, val_end)
    predictions["ano_anterior"] = pd.Series(
        [fill(dev.iloc[:o]).iloc[o - 36] for o in range(train_end, val_end)], index=val.index)
    return {"dev": dev, "val": val, "scale": scale, "predictions": predictions,
            "dev_rows": dev_rows, "train_rows": train_rows, "folds": folds}


def block_bootstrap_diff(y_true, pred, ref, scale, block=6, n=5000, seed=0):
    """IC percentil e proporção de reamostragens com diferença de MASE negativa."""
    observed_errors(y_true, pred)
    observed_errors(y_true, ref)
    if not 1 <= block <= len(y_true) or n < 1:
        raise ValueError("Bloco e número de reamostragens inválidos.")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("A escala do treinamento deve ser finita e positiva.")
    rng = np.random.default_rng(seed)
    ok = y_true.notna().to_numpy()
    e_m = np.abs(y_true - pred).to_numpy() / scale
    e_r = np.abs(y_true - ref).to_numpy() / scale
    T = len(y_true)
    starts = np.arange(T - block + 1)
    diffs = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in
                              rng.choice(starts, size=int(np.ceil(T / block)))])[:T]
        sel = idx[ok[idx]]
        if len(sel):
            diffs.append(e_m[sel].mean() - e_r[sel].mean())
    if not diffs:
        raise ValueError("Nenhuma reamostragem continha um alvo observado.")
    diffs = np.asarray(diffs)
    point = e_m[ok].mean() - e_r[ok].mean()
    return point, *np.percentile(diffs, [2.5, 97.5]), (diffs < 0).mean()
