from __future__ import annotations
import numpy as np, pandas as pd


def _normalize_leg(g, weight_col, sign=1.0):
    w=g[weight_col].clip(lower=0).to_numpy(float)
    if w.sum()<=0: w=np.ones(len(g),float)
    w=sign*w/w.sum()
    return pd.Series(w,index=g.index)


def portfolio_holdings(pred: pd.DataFrame,n_quantiles=10,weight_col="market_cap"):
    """Target holdings at each month-end for D10 and D10-D1 value-weighted portfolios."""
    d=pred.copy(); d["date"]=pd.to_datetime(d["date"])
    d["decile"]=d.groupby("date")["prediction"].transform(lambda s:pd.qcut(s.rank(method="first"),n_quantiles,labels=False,duplicates="drop")+1)
    d["w_long_only"]=0.0; d["w_long_short"]=0.0
    for _,g in d.groupby("date"):
        top=g[g.decile==n_quantiles]; bot=g[g.decile==1]
        d.loc[top.index,"w_long_only"]=_normalize_leg(top,weight_col,+1)
        d.loc[top.index,"w_long_short"]=_normalize_leg(top,weight_col,+1)
        d.loc[bot.index,"w_long_short"]=_normalize_leg(bot,weight_col,-1)
    return d


def portfolio_returns(pred: pd.DataFrame,n_quantiles=10,weight_col="market_cap",ret_col="target"):
    h=portfolio_holdings(pred,n_quantiles,weight_col)
    h["lo_contrib"]=h.w_long_only*h[ret_col]
    h["ls_contrib"]=h.w_long_short*h[ret_col]
    return h.groupby("date",as_index=False).agg(long_only=("lo_contrib","sum"),long_short=("ls_contrib","sum")).sort_values("date")
