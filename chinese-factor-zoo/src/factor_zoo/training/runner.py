from __future__ import annotations

import numpy as np
import pandas as pd

from factor_zoo.training.splits import expanding_yearly_splits
from factor_zoo.training.tuning import tune_model
from factor_zoo.models.registry import build_model
from factor_zoo.models.neural import NeuralRegressor
from factor_zoo.models.ensemble import AveragingEnsemble


def run_expanding_experiment(
    panel: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    model_cfg: dict,
    experiment_cfg: dict,
    date_col: str = "date",
    target_col: str = "target",
    stock_col: str = "stock_id",
    max_candidates=None,
):
    """Run the paper-style yearly expanding experiment efficiently.

    The original version sliced the full 1,160-column DataFrame for every yearly
    window. That was correct but unnecessarily slow. Here we materialize only the
    columns required by the current model once, then use NumPy row indices.
    """
    rows: list[pd.DataFrame] = []
    chosen: list[dict] = []
    exp = experiment_cfg["monthly"]

    cols = model_cfg["models"][model_name].get("features", feature_cols)
    X_all = panel[cols].to_numpy(dtype=np.float32, copy=True)
    y_all = panel[target_col].to_numpy(dtype=np.float32, copy=True)
    dates = pd.to_datetime(panel[date_col]).to_numpy()

    meta_cols = [stock_col, date_col, target_col]
    if "future_return" in panel.columns:
        meta_cols.append("future_return")
    for extra in ["market_cap", "soe", "avg_mcap_per_shareholder"]:
        if extra in panel.columns:
            meta_cols.append(extra)

    for split in expanding_yearly_splits(
        exp["train_start"],
        exp["first_train_end"],
        exp["validation_years"],
        exp["test_years"],
        exp["final_test_end"],
    ):
        tr_idx = np.flatnonzero((dates >= np.datetime64(split.train_start)) & (dates <= np.datetime64(split.train_end)))
        va_idx = np.flatnonzero((dates >= np.datetime64(split.val_start)) & (dates <= np.datetime64(split.val_end)))
        te_idx = np.flatnonzero((dates >= np.datetime64(split.test_start)) & (dates <= np.datetime64(split.test_end)))
        if len(tr_idx) == 0 or len(va_idx) == 0 or len(te_idx) == 0:
            continue

        Xtr, ytr = X_all[tr_idx], y_all[tr_idx]
        Xv, yv = X_all[va_idx], y_all[va_idx]

        # OLS variants have no tuning hyperparameter. Avoid an unnecessary extra fit.
        if model_name in {"ols_h", "ols3_h"}:
            params = {}
        else:
            _, params, _ = tune_model(
                model_name, model_cfg, Xtr, ytr, Xv, yv, max_candidates=max_candidates
            )

        cfg = model_cfg["models"][model_name]
        if cfg["family"] == "neural_net":
            d = model_cfg["neural_defaults"]
            base_seed = model_cfg.get("seed", 42)

            def factory(i):
                return NeuralRegressor(
                    cfg["hidden_layers"],
                    M=model_cfg.get("huber_M", 1.35),
                    l1=params.get("l1", 1e-4),
                    lr=params.get("lr", 1e-3),
                    batch_size=params.get("batch_size", 2048),
                    epochs=d["epochs"],
                    patience=d["patience"],
                    dropout=params.get("dropout", d.get("dropout", 0.2)),
                    batch_norm=d["batch_norm"],
                    seed=base_seed + i,
                )

            model = AveragingEnsemble(factory, n_models=d.get("ensemble", 10))
        else:
            model = build_model(
                model_name,
                model_cfg,
                params,
                seed=model_cfg.get("seed", 42),
                M=model_cfg.get("huber_M", 1.35),
            )

        if model_name in {"ols_h", "ols3_h"}:
            fit_idx = np.concatenate([tr_idx, va_idx])
            model.fit(X_all[fit_idx], y_all[fit_idx])
        else:
            try:
                model.fit(Xtr, ytr, Xv, yv)
            except TypeError:
                model.fit(Xtr, ytr)

        pred = np.asarray(model.predict(X_all[te_idx])).ravel()
        out = panel.iloc[te_idx][meta_cols].copy()
        out["prediction"] = pred
        out["model"] = model_name
        rows.append(out)
        chosen.append({"test_start": str(split.test_start.date()), "params": params})

    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), chosen
