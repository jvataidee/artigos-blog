"""Regressões para o isolamento temporal e a avaliação em alvos observados."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from protocolo import (
    block_bootstrap_diff,
    load_series,
    make_features_age,
    mase,
    new_model,
    row_for_day,
    run_development,
    walk_forward,
)

HERE = Path(__file__).parent


class ProtocoloTest(unittest.TestCase):
    def test_alvo_e_futuro_nao_alteram_entrada_causal(self):
        idx = pd.date_range("2020-01-01", periods=12, freq="10D")
        s = pd.Series([0.3, 0.4, np.nan, 0.7, np.nan, 0.9, 0.5, np.nan, 0.3, 0.4, 0.5, 0.6], index=idx)
        changed = s.copy()
        changed.iloc[6:] = 99
        pd.testing.assert_frame_equal(row_for_day(s, 6, make_features_age),
                                      row_for_day(changed, 6, make_features_age))

    def test_2025_nao_altera_desenvolvimento(self):
        _, _, series = load_series(HERE / "dados" / "pixel_soja_s2.csv")
        original = run_development(series, 144, 216, n_estimators=3)
        changed = series.copy()
        changed.iloc[216:] = 99  # muda os valores e a máscara de observabilidade
        altered = run_development(changed, 144, 216, n_estimators=3)
        self.assertEqual(original["scale"], altered["scale"])
        for key in original["predictions"]:
            np.testing.assert_allclose(original["predictions"][key], altered["predictions"][key],
                                       rtol=0, atol=1e-12, equal_nan=True)

    def test_previsao_em_lote_equivale_as_origens_separadas(self):
        _, _, s = load_series(HERE / "dados" / "pixel_soja_s2.csv")
        rows = [i for i in range(4, 144) if pd.notna(s.iloc[i])]
        X = pd.concat([row_for_day(s, i) for i in rows])
        model = new_model(n_estimators=3).fit(X, s.iloc[rows])
        separate = [model.predict(row_for_day(s, o))[0] for o in range(144, 150)]
        batch = walk_forward(s, 144, 150, n_estimators=3)
        np.testing.assert_allclose(batch, separate, atol=1e-12, rtol=0)

    def test_mase_nao_descarta_previsao_ausente_em_alvo_observado(self):
        y = pd.Series([1.0, np.nan, 2.0])
        self.assertEqual(mase(y, pd.Series([0.0, 999.0, 1.0]), 2), 0.5)
        with self.assertRaises(ValueError):
            mase(y, pd.Series([0.0, 0.0, np.nan]), 2)

    def test_bootstrap_pareado_identico_tem_diferenca_zero(self):
        y = pd.Series([0.3, np.nan, 0.8, 0.5, 0.4, 0.2])
        p = pd.Series([0.2] * 6)
        self.assertEqual(block_bootstrap_diff(y, p, p, 0.1, block=3, n=100), (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
