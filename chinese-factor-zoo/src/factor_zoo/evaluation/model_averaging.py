from __future__ import annotations
import pandas as pd


def average_prediction_files(frames: dict[str,pd.DataFrame], keys=("stock_id","date")) -> pd.DataFrame:
    """Equal-weight forecast combination, useful as a paper extension/robustness check."""
    merged=None; pred_cols=[]
    for name,df in frames.items():
        z=df.copy().rename(columns={"prediction":f"prediction__{name}"})
        keep=list(keys)+[f"prediction__{name}"]
        for c in ["target","future_return","market_cap","soe","avg_mcap_per_shareholder"]:
            if c in z and (merged is None or c not in merged): keep.append(c)
        merged=z[keep] if merged is None else merged.merge(z[list(keys)+[f"prediction__{name}"]],on=list(keys),how="inner")
        pred_cols.append(f"prediction__{name}")
    merged["prediction"]=merged[pred_cols].mean(axis=1); merged["model"]="model_average"
    return merged
