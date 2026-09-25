from __future__ import annotations
import numpy as np
import pandas as pd


def cross_sectional_rank_to_unit_interval(df: pd.DataFrame, columns: list[str], date_col: str = "date") -> pd.DataFrame:
    """Rank continuous characteristics each month and map ranks to [-1, 1]. Missing -> 0.

    This is the paper's stock-characteristic normalization. Ties use average ranks.
    """
    out = df.copy()
    def _rank(g: pd.DataFrame) -> pd.DataFrame:
        ranked = g[columns].rank(method="average", pct=True)
        ranked = 2.0 * ranked - 1.0
        return ranked.fillna(0.0)
    out[columns] = out.groupby(date_col, group_keys=False)[columns].apply(_rank)
    return out


def make_industry_dummies(df: pd.DataFrame, industry_col: str = "industry", expected: int | None = 80) -> tuple[pd.DataFrame, list[str]]:
    d = pd.get_dummies(df[industry_col].astype("string"), prefix="ind", dtype=float)
    if expected is not None and d.shape[1] != expected:
        # Do not fabricate sectors; warn via attribute and keep observed dummies.
        d.attrs["warning"] = f"Observed {d.shape[1]} industry categories; paper uses {expected}."
    out = pd.concat([df.reset_index(drop=True), d.reset_index(drop=True)], axis=1)
    return out, list(d.columns)


def add_macro_interactions(df: pd.DataFrame, stock_cols: list[str], macro_cols: list[str], macro_prefix: str = "macro__") -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    block = {}
    names: list[str] = []
    for m in macro_cols:
        source = f"{macro_prefix}{m}" if f"{macro_prefix}{m}" in out.columns else m
        for c in stock_cols:
            name = f"{c}__x__{m}"
            block[name] = out[c].to_numpy(dtype=float) * out[source].to_numpy(dtype=float)
            names.append(name)
    out = pd.concat([out, pd.DataFrame(block, index=out.index)], axis=1)
    return out, names
