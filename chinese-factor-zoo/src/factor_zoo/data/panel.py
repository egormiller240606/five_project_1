from __future__ import annotations
import pandas as pd
from factor_zoo.data.validation import require_columns, assert_unique_panel
from factor_zoo.features.catalog import STOCK_CHARACTERISTICS, MACRO_VARIABLES
from factor_zoo.features.preprocessing import cross_sectional_rank_to_unit_interval, make_industry_dummies, add_macro_interactions


def build_model_panel(characteristics: pd.DataFrame, macro: pd.DataFrame, returns: pd.DataFrame,
                      stock_col="stock_id", date_col="date", return_col="return", rf_col="risk_free",
                      industry_col="industry", expected_industries=80) -> tuple[pd.DataFrame, list[str]]:
    require_columns(characteristics, [stock_col, date_col, industry_col] + STOCK_CHARACTERISTICS, "characteristics")
    require_columns(macro, [date_col] + MACRO_VARIABLES, "macro")
    require_columns(returns, [stock_col, date_col, return_col, rf_col], "returns")
    assert_unique_panel(characteristics, stock_col, date_col)
    assert_unique_panel(returns, stock_col, date_col)

    c = cross_sectional_rank_to_unit_interval(characteristics, STOCK_CHARACTERISTICS, date_col)
    macro_block = macro[[date_col] + MACRO_VARIABLES].rename(columns={m: f"macro__{m}" for m in MACRO_VARIABLES})
    x = c.merge(macro_block, on=date_col, how="left", validate="many_to_one")
    x, interaction_cols = add_macro_interactions(x, STOCK_CHARACTERISTICS, MACRO_VARIABLES)
    x, industry_cols = make_industry_dummies(x, industry_col, expected_industries)

    # target: information at t predicts excess return at t+1
    r = returns[[stock_col, date_col, return_col, rf_col]].copy().sort_values([stock_col, date_col])
    r["excess_return"] = r[return_col] - r[rf_col]
    r["target"] = r.groupby(stock_col)["excess_return"].shift(-1)
    r["future_return"] = r.groupby(stock_col)[return_col].shift(-1)
    panel = x.merge(r[[stock_col, date_col, "target", "future_return"]], on=[stock_col, date_col], how="left", validate="one_to_one")
    panel = panel.dropna(subset=["target"]).reset_index(drop=True)
    feature_cols = STOCK_CHARACTERISTICS + interaction_cols + industry_cols
    return panel, feature_cols
