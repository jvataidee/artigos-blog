"""Comparações pareadas, sementes, sensibilidade do bootstrap e avaliação descritiva do teste."""
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from figura_cubo import load_cube
from protocolo import (
    SEEDS,
    block_bootstrap_diff,
    load_series,
    make_features_age,
    mase,
    persistence,
    run_development,
    walk_forward,
)

HERE = Path(__file__).parent


def bootstrap_rows(y, predictions, ref, scale, period):
    rows = []
    for block in (3, 6, 12):
        for name, pred in predictions.items():
            d, lo, hi, fraction = block_bootstrap_diff(y, pred, ref, scale, block=block)
            rows.append(dict(periodo=period, modelo=name, bloco=block, diferenca=d,
                             inferior=lo, superior=hi, proporcao_negativa=fraction))
    return rows


def main():
    out = HERE / "resultados"
    out.mkdir(exist_ok=True)
    _, _, series = load_series(HERE / "dados" / "pixel_soja_s2.csv")
    train_end = series.index.get_loc(pd.Timestamp("2023-01-01"))
    val_end = series.index.get_loc(pd.Timestamp("2025-01-01"))
    # O desenvolvimento recebe apenas o período anterior ao teste.
    dev = series.iloc[:val_end].copy()
    result = run_development(dev, train_end, val_end)
    val, scale, p = result["val"], result["scale"], result["predictions"]
    metrics = pd.DataFrame([dict(modelo=k, MASE=mase(val, v, scale)) for k, v in p.items()])
    metrics.to_csv(out / "validacao.csv", index=False)
    print("MASE na validação (semente 42):\n", metrics.round(3).to_string(index=False), flush=True)

    seeds = []
    for seed in SEEDS:
        r = run_development(dev, train_end, val_end, seed=seed)
        seeds.append({"semente": seed, **{k: mase(val, v, scale) for k, v in r["predictions"].items()
                                         if k not in ("persistencia", "ano_anterior")}})
        print(f"semente de desenvolvimento concluída: {seed}", flush=True)
    seeds = pd.DataFrame(seeds).set_index("semente")
    seeds.to_csv(out / "sementes_validacao.csv")
    summary = seeds.agg(["min", "median", "max"]).T
    summary.to_csv(out / "resumo_sementes.csv", index_label="modelo")
    print("\nSementes 0 a 9 (modelo e dobras; comparação pareada):\n", summary.round(3).to_string(), flush=True)

    boot = bootstrap_rows(val, {k: p[k] for k in ("kfold", "kfold_passado", "pre", "wf", "idade")},
                          p["persistencia"], scale, "validacao")
    print("\nBootstrap da validação:\n", pd.DataFrame(boot).round(3).to_string(index=False), flush=True)
    safra = val.index.month.isin([10, 11, 12, 1, 2])
    phases = []
    for key in ("persistencia", "pre", "wf", "idade"):
        for label, mask in [("safra", safra), ("resto", ~safra)]:
            phases.append(dict(modelo=key, fase=label, n=int(val[mask].notna().sum()),
                               MASE=mase(val[mask], p[key][mask], scale)))
    pd.DataFrame(phases).to_csv(out / "fases.csv", index=False)
    print("\nFases:\n", pd.DataFrame(phases).round(3).to_string(index=False), flush=True)

    grid, cube = load_cube(until="2025-01-01")
    center = cube[:, 16, 16]
    corr = np.full((32, 32), np.nan)
    for r in range(32):
        for c in range(32):
            s = cube[:, r, c]
            ok = np.isfinite(s) & np.isfinite(center)
            if ok.sum() > 30 and np.std(s[ok]) > 0 and np.std(center[ok]) > 0:
                corr[r, c] = np.corrcoef(s[ok], center[ok])[0, 1]
    cube_stats = {"periodo": "2019 a 2024", "decendios": len(grid),
                  "pixels_com_correlacao": int(np.isfinite(corr).sum()),
                  "mediana": float(np.nanmedian(corr)),
                  "fracao_1024_acima_09": float(np.mean(corr > 0.9))}
    print("\nCubo até 2024:", cube_stats, flush=True)

    selected = min(("persistencia", "wf", "idade"), key=lambda key: mase(val, p[key], scale))
    print("\nDecisão anterior ao teste:", selected, flush=True)
    # A partir daqui 2025 é usado apenas para relatar o desempenho, sem rever a seleção.
    sealed = series.iloc[val_end:]
    ref = persistence(series, val_end, len(series))
    pred_test = walk_forward(series, val_end, len(series), features=make_features_age)
    boot.extend(bootstrap_rows(sealed, {"idade": pred_test}, ref, scale, "teste_descritivo"))
    pd.DataFrame(boot).to_csv(out / "bootstrap.csv", index=False)
    test_seeds = pd.DataFrame([dict(semente=seed, MASE=mase(sealed,
        walk_forward(series, val_end, len(series), features=make_features_age, seed=seed), scale))
        for seed in SEEDS])
    test_seeds.to_csv(out / "sementes_teste_descritivo.csv", index=False)
    test_metrics = {"persistencia": float(mase(sealed, ref, scale)),
                    "idade": float(mase(sealed, pred_test, scale))}
    print("\nTeste descritivo:", test_metrics)
    print(pd.DataFrame(boot).query("periodo == 'teste_descritivo'").round(3).to_string(index=False))

    # Conferência estrutural: valores e máscara de 2025 não podem alterar o desenvolvimento.
    altered = series.copy()
    altered.iloc[val_end:] = 99.0
    check = run_development(altered, train_end, val_end)
    for key in p:
        np.testing.assert_allclose(p[key], check["predictions"][key], atol=1e-12, rtol=0, equal_nan=True)
    assert scale == check["scale"]
    provenance = {"python": platform.python_version(), "numpy": np.__version__,
                  "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                  "matplotlib": matplotlib.__version__,
                  "semente_principal": 42, "sementes_sensibilidade": list(SEEDS),
                  "blocos": [3, 6, 12], "reamostragens": 5000, "semente_bootstrap": 0,
                  "escala_treino": float(scale), "selecionado": selected,
                  "teste_descritivo": test_metrics, "cubo": cube_stats,
                  "invariancia_a_2025": True, "linhas_desenvolvimento": result["dev_rows"],
                  "linhas_treino": result["train_rows"],
                  "sha256_codigo": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                     for f in sorted(HERE.glob("*.py"))},
                  "sha256_dados": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                    for f in sorted((HERE / "dados").iterdir()) if f.is_file()}}
    (out / "proveniencia.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nConferência aprovada: alterar todo o teste não muda nenhuma previsão de desenvolvimento.")


if __name__ == "__main__":
    main()
