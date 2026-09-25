from __future__ import annotations
import pandas as pd


def exclude_price_limit_buys(pred: pd.DataFrame, upper_limit_col="at_upper_limit") -> pd.DataFrame:
    """Robustness hook for the paper's price-limit exercise.

    If the vendor export marks stocks closed at the upper daily price limit on a
    rebalancing date, they are excluded from new buying targets.
    """
    if upper_limit_col not in pred: return pred.copy()
    return pred.loc[~pred[upper_limit_col].fillna(False)].copy()


def top70_by_market_cap(pred: pd.DataFrame) -> pd.DataFrame:
    q=pred.groupby("date")["market_cap"].transform(lambda x:x.quantile(.30))
    return pred.loc[pred.market_cap>q].copy()
