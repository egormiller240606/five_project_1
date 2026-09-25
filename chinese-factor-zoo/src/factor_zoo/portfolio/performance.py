from __future__ import annotations
import numpy as np, pandas as pd


def stats(r):
    """Paper-style monthly portfolio statistics; return moments remain in decimal units."""
    x=pd.Series(r,dtype=float).dropna(); sd=x.std(ddof=1)
    wealth=(1+x).cumprod(); drawdown=wealth/wealth.cummax()-1
    return {
        "avg_monthly":x.mean(),
        "std_monthly":sd,
        "sharpe_ann":np.sqrt(12)*x.mean()/sd if sd else np.nan,
        "skew":x.skew(),
        "kurtosis":x.kurt(),
        "max_drawdown":float(-drawdown.min()),
        "max_1m_loss":float(-x.min()),
    }
