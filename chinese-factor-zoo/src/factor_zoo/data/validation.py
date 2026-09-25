from __future__ import annotations
import pandas as pd


def require_columns(df: pd.DataFrame, cols: list[str], label: str = "dataframe") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def assert_unique_panel(df: pd.DataFrame, stock_col: str = "stock_id", date_col: str = "date") -> None:
    dup = df.duplicated([stock_col, date_col])
    if dup.any():
        ex = df.loc[dup, [stock_col, date_col]].head().to_dict("records")
        raise ValueError(f"Duplicate stock-month rows found, examples: {ex}")
