from __future__ import annotations
import numpy as np, pandas as pd
from factor_zoo.evaluation.metrics import r2_oos

def zero_out_importance(model, X, y, feature_names):
    X=np.asarray(X); base=r2_oos(y,model.predict(X)); vals=[]
    for j,name in enumerate(feature_names):
        z=X.copy(); z[:,j]=0.0; vals.append((name,base-r2_oos(y,model.predict(z))))
    ans=pd.DataFrame(vals,columns=["feature","importance"])
    s=ans.importance.clip(lower=0).sum()
    ans["relative_importance"] = ans.importance.clip(lower=0)/s if s else 0.0
    return ans.sort_values("importance",ascending=False)
