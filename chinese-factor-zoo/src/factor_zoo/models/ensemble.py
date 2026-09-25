from __future__ import annotations
import numpy as np

class AveragingEnsemble:
    def __init__(self, factory, n_models=10): self.factory=factory; self.n_models=n_models
    def fit(self,X,y,X_val,y_val):
        self.models=[]
        for i in range(self.n_models):
            m=self.factory(i); m.fit(X,y,X_val,y_val); self.models.append(m)
        return self
    def predict(self,X): return np.mean([m.predict(X) for m in self.models],axis=0)
