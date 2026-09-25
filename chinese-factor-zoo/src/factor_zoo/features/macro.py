from __future__ import annotations
import numpy as np
import pandas as pd


def safe_log(x):
    x=pd.Series(x,dtype=float)
    return np.log(x.where(x>0))


def build_macro_predictors(df: pd.DataFrame, date_col="date") -> pd.DataFrame:
    """Build the paper's 11 macro predictors from canonical aggregate fields.

    Expected raw field names when a predictor is not already supplied:
      dividends_12m, weighted_price, earnings, aggregate_book, aggregate_market,
      sse_daily_return_sq_sum, net_equity_issues_12m, a_share_market_cap,
      gov10y_yield, gov1y_yield, cpi, market_turnover, m2, trade_volume.

    The function is intentionally field-code agnostic: WIND/CSMAR export codes vary.
    """
    x=df.copy().sort_values(date_col); x[date_col]=pd.to_datetime(x[date_col])
    if "dp" not in x and {"dividends_12m","weighted_price"} <= set(x): x["dp"]=safe_log(x.dividends_12m)-safe_log(x.weighted_price)
    if "de" not in x and {"dividends_12m","earnings"} <= set(x): x["de"]=safe_log(x.dividends_12m)-safe_log(x.earnings)
    if "bm" not in x and {"aggregate_book","aggregate_market"} <= set(x): x["bm"]=x.aggregate_book/x.aggregate_market.replace(0,np.nan)
    if "svar" not in x and "sse_daily_return_sq_sum" in x: x["svar"]=x.sse_daily_return_sq_sum
    if "ep" not in x and {"weighted_eps","weighted_price"} <= set(x): x["ep"]=safe_log(x.weighted_eps)-safe_log(x.weighted_price)
    if "ntis" not in x and {"net_equity_issues_12m","a_share_market_cap"} <= set(x): x["ntis"]=x.net_equity_issues_12m/x.a_share_market_cap.replace(0,np.nan)
    if "tms" not in x and {"gov10y_yield","gov1y_yield"} <= set(x): x["tms"]=x.gov10y_yield-x.gov1y_yield
    if "infl" not in x and "cpi" in x: x["infl"]=x.cpi.pct_change()
    if "mtr" not in x and "market_turnover" in x: x["mtr"]=x.market_turnover
    if "m2gr" not in x and "m2" in x: x["m2gr"]=x.m2.pct_change(12)
    if "itgr" not in x and "trade_volume" in x: x["itgr"]=x.trade_volume.pct_change(12)
    return x
