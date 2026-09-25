from __future__ import annotations
import numpy as np, pandas as pd


def _drift_weights(prev: pd.DataFrame, weight_col: str, ret_col: str):
    w=prev[weight_col].to_numpy(float); r=prev[ret_col].fillna(0).to_numpy(float)
    out=w*(1+r)
    # Preserve the target gross exposure separately for positive/negative legs.
    pos=out>0; neg=out<0
    if pos.any() and out[pos].sum()!=0: out[pos] /= out[pos].sum()
    if neg.any() and -out[neg].sum()!=0: out[neg] /= -out[neg].sum()
    return pd.Series(out,index=prev.index)


def holdings_turnover(holdings: pd.DataFrame, weight_col: str, stock_col="stock_id", date_col="date", ret_col="future_return") -> pd.Series:
    """Half-L1 turnover between target weights and drifted previous holdings.

    This is the standard one-way turnover convention: 0.5*sum(|w_t - w^-_t|).
    """
    h=holdings.copy(); h[date_col]=pd.to_datetime(h[date_col]); dates=sorted(h[date_col].unique())
    vals={}; prev=None
    for dt in dates:
        cur=h[h[date_col]==dt].set_index(stock_col)
        if prev is None:
            vals[dt]=0.5*np.abs(cur[weight_col]).sum()
        else:
            pw=_drift_weights(prev,weight_col,ret_col); cw=cur[weight_col]
            ids=pw.index.union(cw.index)
            vals[dt]=0.5*np.abs(cw.reindex(ids,fill_value=0)-pw.reindex(ids,fill_value=0)).sum()
        prev=cur
    return pd.Series(vals,name=f"turnover_{weight_col}")


def apply_turnover_costs(returns: pd.DataFrame, holdings: pd.DataFrame, bps:int, date_col="date"):
    out=returns.copy(); c=bps/10000.0
    for strategy,wcol in [("long_only","w_long_only"),("long_short","w_long_short")]:
        t=holdings_turnover(holdings,wcol).rename("turnover").rename_axis(date_col).reset_index()
        out=out.merge(t,on=date_col,how="left",suffixes=("",f"_{strategy}"))
        turn_col="turnover" if strategy=="long_only" else f"turnover_{strategy}"
        out[f"{strategy}_net_{bps}bps"]=out[strategy]-c*out[turn_col]
    return out


def apply_simple_round_trip_cost(portfolio_returns: pd.DataFrame,bps:int,columns=("long_only","long_short")):
    """Deprecated compatibility helper. Prefer `apply_turnover_costs`."""
    out=portfolio_returns.copy(); c=bps/10000.0
    for col in columns: out[f"{col}_net_{bps}bps"]=out[col]-c
    return out
