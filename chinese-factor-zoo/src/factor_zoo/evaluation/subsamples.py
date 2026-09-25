from __future__ import annotations
import pandas as pd
from factor_zoo.evaluation.metrics import r2_oos

def evaluate_subsamples(pred: pd.DataFrame):
    out={"all":r2_oos(pred.target,pred.prediction)}
    if "soe" in pred:
        out["soe"]=r2_oos(pred.loc[pred.soe==1,"target"],pred.loc[pred.soe==1,"prediction"])
        out["non_soe"]=r2_oos(pred.loc[pred.soe==0,"target"],pred.loc[pred.soe==0,"prediction"])
    if "market_cap" in pred:
        q=pred.groupby("date")["market_cap"].transform(lambda s:s.quantile(.30))
        out["bottom_30_mcap"]=r2_oos(pred.loc[pred.market_cap<=q,"target"],pred.loc[pred.market_cap<=q,"prediction"])
        out["top_70_mcap"]=r2_oos(pred.loc[pred.market_cap>q,"target"],pred.loc[pred.market_cap>q,"prediction"])
    if "avg_mcap_per_shareholder" in pred:
        q=pred.groupby("date")["avg_mcap_per_shareholder"].transform(lambda s:s.quantile(.30))
        out["bottom_30_shareholder"]=r2_oos(pred.loc[pred.avg_mcap_per_shareholder<=q,"target"],pred.loc[pred.avg_mcap_per_shareholder<=q,"prediction"])
        out["top_70_shareholder"]=r2_oos(pred.loc[pred.avg_mcap_per_shareholder>q,"target"],pred.loc[pred.avg_mcap_per_shareholder>q,"prediction"])
    return out
