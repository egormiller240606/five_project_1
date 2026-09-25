from __future__ import annotations

"""Structured synthetic Chinese A-share style panel for pipeline validation.

The generator intentionally mimics the *shape* of the paper's data (monthly panel,
90 stock characteristics, 11 macro predictors, industries, size and SOE flags),
but it does not attempt to recreate the proprietary WIND/CSMAR observations or
force the paper's reported results. Returns are generated from a known nonlinear
signal plus noise so that the replication pipeline has something real to learn.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from factor_zoo.features.catalog import (
    CHARACTERISTIC_GROUPS,
    FREQUENCY,
    MACRO_VARIABLES,
    STOCK_CHARACTERISTICS,
)


@dataclass(frozen=True)
class SyntheticSpec:
    n_stocks: int = 120
    start: str = "2000-01-31"
    end: str = "2020-06-30"
    seed: int = 2026
    n_industries: int = 80
    signal_monthly_std: float = 0.012
    idiosyncratic_monthly_std: float = 0.055
    market_monthly_std: float = 0.020


def _ar1(rng: np.random.Generator, n_t: int, n_n: int, rho: float = 0.82) -> np.ndarray:
    eps = rng.normal(size=(n_t, n_n))
    out = np.empty_like(eps)
    out[0] = eps[0]
    scale = np.sqrt(max(1.0 - rho * rho, 1e-8))
    for t in range(1, n_t):
        out[t] = rho * out[t - 1] + scale * eps[t]
    return out


def _hold_frequency(values: np.ndarray, frequency: str) -> np.ndarray:
    """Make lower-frequency characteristics change only at their update months."""
    step = {"monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12}.get(frequency, 1)
    if step == 1:
        return values
    out = values.copy()
    for t in range(values.shape[0]):
        anchor = (t // step) * step
        out[t] = values[anchor]
    return out


def _rank_minus1_plus1(values: np.ndarray) -> np.ndarray:
    """Cross-sectional rank each month, matching the downstream normalization idea."""
    frame = pd.DataFrame(values)
    return (2.0 * frame.rank(axis=1, method="average", pct=True) - 1.0).to_numpy(float)


def _group_for_characteristic(name: str) -> str:
    for group, names in CHARACTERISTIC_GROUPS.items():
        if name in names:
            return group
    return "misc"


def _write_portable(frame: pd.DataFrame, base: Path) -> str:
    """Prefer Parquet, fall back to CSV when pyarrow is unavailable."""
    try:
        frame.to_parquet(base.with_suffix(".parquet"), index=False)
        return "parquet"
    except (ImportError, ModuleNotFoundError):
        frame.to_csv(base.with_suffix(".csv"), index=False)
        return "csv"


def generate_synthetic_raw(output_dir: str | Path, spec: SyntheticSpec = SyntheticSpec()) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)

    dates = pd.date_range(spec.start, spec.end, freq="ME")
    n_t, n = len(dates), int(spec.n_stocks)
    if n < 20:
        raise ValueError("Use at least 20 stocks so decile portfolios are meaningful.")
    stocks = np.array([f"S{i:04d}" for i in range(n)])

    # 11 persistent aggregate predictors. Values are standardized for the return DGP,
    # while the raw series are written to disk and later consumed by the normal pipeline.
    macro_raw = {}
    for j, name in enumerate(MACRO_VARIABLES):
        rho = 0.70 + 0.02 * (j % 6)
        macro_raw[name] = _ar1(rng, n_t, 1, rho=rho).ravel()
    macro = pd.DataFrame({"date": dates, **macro_raw})
    macro_z = {
        k: (v - np.mean(v)) / (np.std(v) + 1e-12)
        for k, v in macro_raw.items()
    }

    # Persistent stock-level latent states. These create correlated characteristics
    # instead of 90 unrelated white-noise columns.
    groups = [
        "size", "beta", "momentum", "liquidity", "volatility", "ownership",
        "book_to_price", "earnings", "growth", "leverage", "misc",
    ]
    latent = {}
    stock_loading = {g: rng.normal(size=n) for g in groups}
    common = {g: _ar1(rng, n_t, 1, rho=0.90).ravel() for g in groups}
    for g in groups:
        idio = _ar1(rng, n_t, n, rho=0.78)
        latent[g] = (
            0.55 * stock_loading[g][None, :]
            + 0.20 * common[g][:, None]
            + 0.70 * idio
        )

    char_arrays: dict[str, np.ndarray] = {}
    for j, name in enumerate(STOCK_CHARACTERISTICS):
        group = _group_for_characteristic(name)
        base = latent[group]
        # A characteristic-specific loading keeps variables within a category related,
        # but far from identical.
        sign = -1.0 if name in {"ill", "zerotrade", "pricedelay", "lev"} else 1.0
        noise = _ar1(rng, n_t, n, rho=0.50 + 0.05 * (j % 5))
        values = sign * base + 0.45 * noise + 0.08 * rng.normal(size=(1, n))
        values = _hold_frequency(values, FREQUENCY.get(name, "quarterly"))
        char_arrays[name] = values.astype(np.float32)

    # Make economically interpretable anchors more directly tied to the latent states.
    char_arrays["mve"] = _hold_frequency(latent["size"] + 0.15 * _ar1(rng, n_t, n), "monthly").astype(np.float32)
    char_arrays["bm"] = _hold_frequency(latent["book_to_price"] + 0.15 * _ar1(rng, n_t, n), "quarterly").astype(np.float32)
    char_arrays["mom1m"] = (latent["momentum"] + 0.20 * _ar1(rng, n_t, n)).astype(np.float32)
    char_arrays["mom6m"] = (0.85 * latent["momentum"] + 0.25 * _ar1(rng, n_t, n)).astype(np.float32)
    char_arrays["mom12m"] = (0.75 * latent["momentum"] + 0.30 * _ar1(rng, n_t, n)).astype(np.float32)
    char_arrays["turn"] = (latent["liquidity"] + 0.20 * _ar1(rng, n_t, n)).astype(np.float32)
    char_arrays["atr"] = (0.8 * latent["liquidity"] + 0.35 * _ar1(rng, n_t, n)).astype(np.float32)
    char_arrays["volatility"] = (latent["volatility"] + 0.20 * _ar1(rng, n_t, n)).astype(np.float32)
    char_arrays["roaq"] = _hold_frequency(latent["earnings"] + 0.20 * _ar1(rng, n_t, n), "quarterly").astype(np.float32)
    char_arrays["largestholderrate"] = _hold_frequency(latent["ownership"] + 0.15 * _ar1(rng, n_t, n), "annual").astype(np.float32)

    # Static identity / industry / ownership variables.
    if n >= spec.n_industries:
        industries = np.arange(n) % spec.n_industries
        rng.shuffle(industries)
    else:
        industries = np.arange(n) % min(spec.n_industries, n)
        rng.shuffle(industries)
    industry_labels = np.array([f"I{x:02d}" for x in industries])

    size_mean = char_arrays["mve"].mean(axis=0)
    own_mean = char_arrays["largestholderrate"].mean(axis=0)
    soe_score = 0.9 * size_mean + 0.35 * own_mean + 0.25 * rng.normal(size=n)
    soe = (soe_score > np.median(soe_score)).astype(np.int8)

    log_mcap = 10.0 + 0.75 * char_arrays["mve"] + 0.08 * np.arange(n_t)[:, None] / max(n_t - 1, 1)
    market_cap = np.exp(log_mcap).astype(np.float64)
    shareholder_count = np.exp(7.5 - 0.15 * own_mean + 0.25 * rng.normal(size=n))
    avg_mcap_per_shareholder = market_cap / shareholder_count[None, :]

    # Known nonlinear return signal based on information at t. Importantly, the
    # downstream model does NOT receive this signal column; it must reconstruct it
    # from the same 1,160 feature construction used for real data.
    r_mom12 = _rank_minus1_plus1(char_arrays["mom12m"])
    r_mom6 = _rank_minus1_plus1(char_arrays["mom6m"])
    r_atr = _rank_minus1_plus1(char_arrays["atr"])
    r_turn = _rank_minus1_plus1(char_arrays["turn"])
    r_vol = _rank_minus1_plus1(char_arrays["volatility"])
    r_bm = _rank_minus1_plus1(char_arrays["bm"])
    r_roaq = _rank_minus1_plus1(char_arrays["roaq"])
    r_mve = _rank_minus1_plus1(char_arrays["mve"])

    raw_signal = (
        0.80 * r_mom12
        + 0.55 * r_bm
        + 0.45 * r_roaq
        + 0.35 * r_turn
        - 0.55 * r_vol
        + 0.70 * r_mom6 * np.tanh(macro_z["tms"])[:, None]
        + 0.55 * r_atr * np.tanh(macro_z["mtr"])[:, None]
        + 0.45 * (r_atr * r_atr - 1.0 / 3.0)
        + 0.35 * np.tanh(2.0 * r_bm * r_roaq)
    )
    # Small/non-SOE stocks have a stronger short-horizon signal, echoing a qualitative
    # pattern studied in the paper without calibrating to its numerical estimates.
    heterogeneity = 1.0 + 0.35 * (r_mve < -0.40) + 0.20 * (1 - soe)[None, :]
    raw_signal *= heterogeneity
    raw_signal -= raw_signal.mean()
    signal = raw_signal * (spec.signal_monthly_std / (raw_signal.std() + 1e-12))

    industry_alpha = rng.normal(0.0, 0.0015, size=spec.n_industries)
    signal += industry_alpha[industries][None, :]

    market_shock = rng.normal(0.0, spec.market_monthly_std, size=n_t)
    idio = rng.normal(0.0, spec.idiosyncratic_monthly_std, size=(n_t, n))
    rf = 0.002 + 0.00025 * np.tanh(macro_z["tms"])
    total_return = np.empty((n_t, n), dtype=float)
    total_return[0] = rf[0] + market_shock[0] + idio[0]
    for t in range(1, n_t):
        total_return[t] = rf[t] + signal[t - 1] + market_shock[t] + idio[t]
    total_return = np.clip(total_return, -0.45, 0.45)

    idx = pd.MultiIndex.from_product([stocks, dates], names=["stock_id", "date"])
    chars = idx.to_frame(index=False)
    for name in STOCK_CHARACTERISTICS:
        chars[name] = char_arrays[name].T.reshape(-1)
    chars["industry"] = np.repeat(industry_labels, n_t)
    chars["market_cap"] = market_cap.T.reshape(-1)
    chars["soe"] = np.repeat(soe, n_t)
    chars["avg_mcap_per_shareholder"] = avg_mcap_per_shareholder.T.reshape(-1)

    rets = idx.to_frame(index=False)
    rets["return"] = total_return.T.reshape(-1)
    rets["risk_free"] = np.tile(rf, n)

    fmt_chars = _write_portable(chars, output_dir / "stock_characteristics")
    fmt_macro = _write_portable(macro, output_dir / "macro")
    fmt_returns = _write_portable(rets, output_dir / "monthly_returns")

    # This is a property of the synthetic DGP, not an empirical result from the paper.
    realized_excess = total_return[1:] - rf[1:, None]
    theoretical_signal = signal[:-1]
    variance_share = float(np.var(theoretical_signal) / max(np.var(realized_excess), 1e-12))
    metadata = {
        **asdict(spec),
        "n_months": n_t,
        "n_rows": int(n_t * n),
        "n_characteristics": len(STOCK_CHARACTERISTICS),
        "n_macro": len(MACRO_VARIABLES),
        "observed_industries": int(len(np.unique(industries))),
        "storage_format": fmt_chars if fmt_chars == fmt_macro == fmt_returns else "mixed",
        "signal_variance_share_approx": variance_share,
        "warning": "Synthetic data validate the pipeline; they are not a replication of WIND/CSMAR empirical estimates.",
    }
    (output_dir / "synthetic_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
