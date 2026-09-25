from __future__ import annotations
import pandas as pd


def asof_merge_financials(monthly: pd.DataFrame, statements: pd.DataFrame, *, stock_col="stock_id",
                          month_col="date", available_col="available_date") -> pd.DataFrame:
    """Point-in-time merge: a statement enters the panel only after it was public.

    `available_date` should be the actual disclosure/announcement date, not fiscal-period end.
    This helper is the main guard against accounting look-ahead bias.
    """
    left=monthly.copy(); right=statements.copy()
    left[month_col]=pd.to_datetime(left[month_col]); right[available_col]=pd.to_datetime(right[available_col])
    left=left.sort_values([month_col,stock_col]); right=right.sort_values([available_col,stock_col])
    return pd.merge_asof(left,right,left_on=month_col,right_on=available_col,by=stock_col,direction="backward",allow_exact_matches=True)
