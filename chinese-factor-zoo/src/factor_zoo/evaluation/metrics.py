from __future__ import annotations
import numpy as np

def r2_oos(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float); p = np.asarray(y_pred, dtype=float)
    den = np.sum(y ** 2)
    return float("nan") if den == 0 else float(1.0 - np.sum((y - p) ** 2) / den)

def mse(y_true, y_pred) -> float:
    y=np.asarray(y_true,float); p=np.asarray(y_pred,float)
    return float(np.mean((y-p)**2))
