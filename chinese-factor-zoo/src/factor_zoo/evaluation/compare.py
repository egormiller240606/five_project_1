from __future__ import annotations
import numpy as np, pandas as pd
from factor_zoo.evaluation.paper_targets import MONTHLY_R2


def compare_monthly_r2(results: pd.DataFrame) -> pd.DataFrame:
    """Compare generated R² (fractions) with paper Table 1 (percent)."""
    rows=[]
    for _,r in results.iterrows():
        model=r.get("model")
        for group,vals in MONTHLY_R2.items():
            if model not in ["ols_h","ols3_h","pls","lasso_h","enet_h","gbrt_h","rf","vasa","nn1","nn2","nn3","nn4","nn5"]: continue
            j=["ols_h","ols3_h","pls","lasso_h","enet_h","gbrt_h","rf","vasa","nn1","nn2","nn3","nn4","nn5"].index(model)
            col={"all":"all","top70":"top_70_mcap","bottom30":"bottom_30_mcap","amcps_top70":"top_70_shareholder","amcps_bottom30":"bottom_30_shareholder","soe":"soe","non_soe":"non_soe"}.get(group)
            if col and col in r and pd.notna(r[col]):
                got=100*float(r[col]); target=float(vals[j]); rows.append({"model":model,"group":group,"paper_pct":target,"replication_pct":got,"abs_diff_pp":abs(got-target)})
    return pd.DataFrame(rows)
