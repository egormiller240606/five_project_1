from pathlib import Path

import pandas as pd

from factor_zoo.data.synthetic import SyntheticSpec, generate_synthetic_raw
from factor_zoo.features.catalog import MACRO_VARIABLES, STOCK_CHARACTERISTICS


def _existing(base: Path) -> Path:
    for suffix in (".parquet", ".csv"):
        p = base.with_suffix(suffix)
        if p.exists():
            return p
    raise AssertionError(f"missing {base}")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def test_synthetic_generator_has_paper_shaped_inputs(tmp_path):
    meta = generate_synthetic_raw(
        tmp_path,
        SyntheticSpec(n_stocks=80, start="2000-01-31", end="2000-03-31", seed=1),
    )
    chars = _read(_existing(tmp_path / "stock_characteristics"))
    macro = _read(_existing(tmp_path / "macro"))
    rets = _read(_existing(tmp_path / "monthly_returns"))

    assert set(STOCK_CHARACTERISTICS).issubset(chars.columns)
    assert set(MACRO_VARIABLES).issubset(macro.columns)
    assert {"stock_id", "date", "return", "risk_free"}.issubset(rets.columns)
    assert chars["industry"].nunique() == 80
    assert meta["n_characteristics"] == 90
    assert meta["n_macro"] == 11
