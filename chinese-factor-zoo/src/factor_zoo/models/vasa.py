from __future__ import annotations
import numpy as np
from sklearn.linear_model import LinearRegression


def marginal_r2_probabilities(X, y, floor=1e-12):
    """Feature-sampling probabilities proportional to univariate in-sample R².

    This implements the importance-weighted predictor subsampling described by the
    VASA appendix/reference rather than uniform feature draws.
    """
    X=np.asarray(X,float); y=np.asarray(y,float).ravel()
    yc=y-np.nanmean(y); vy=np.nansum(yc*yc)
    if not np.isfinite(vy) or vy <= floor:
        return np.full(X.shape[1],1/X.shape[1])
    Xc=X-np.nanmean(X,axis=0)
    num=np.nansum(Xc*yc[:,None],axis=0)**2
    den=np.nansum(Xc*Xc,axis=0)*vy
    r2=np.divide(num,den,out=np.zeros_like(num),where=den>floor)
    r2=np.nan_to_num(r2,nan=0.0,posinf=0.0,neginf=0.0)+floor
    return r2/r2.sum()


class VASARegressor:
    """Variable Subsample Aggregation (clean-room implementation).

    Predictor subsets are sampled without replacement using probabilities
    proportional to marginal in-sample R²; OLS submodels are then averaged.
    """
    def __init__(self,n_submodels=100,n_components=20,seed=42):
        self.n_submodels=int(n_submodels); self.n_components=int(n_components); self.seed=int(seed)
    def fit(self,X,y):
        rng=np.random.default_rng(self.seed); X=np.asarray(X,float); y=np.asarray(y,float).ravel()
        p=X.shape[1]; k=min(self.n_components,p); self.feature_prob_=marginal_r2_probabilities(X,y)
        self.models=[]
        for _ in range(self.n_submodels):
            idx=np.sort(rng.choice(p,size=k,replace=False,p=self.feature_prob_))
            m=LinearRegression().fit(X[:,idx],y); self.models.append((idx,m))
        return self
    def predict(self,X):
        X=np.asarray(X,float)
        return np.mean([m.predict(X[:,idx]) for idx,m in self.models],axis=0)
