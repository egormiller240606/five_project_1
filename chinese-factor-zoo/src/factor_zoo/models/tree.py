from __future__ import annotations
import numpy as np
from sklearn.tree import DecisionTreeRegressor


def paper_huber_numpy(residual, M: float = 1.35):
    r = np.asarray(residual, dtype=float)
    a = np.abs(r)
    return np.where(a <= M, r*r, 2*M*a - M*M)


class PaperHuberGBRT:
    """Clean-room gradient boosting with the paper's fixed Huber threshold M.

    We fit regression trees to clipped residual pseudo-gradients. This preserves the
    paper's fixed Huber transition M=1.35, unlike sklearn's quantile/alpha Huber loss.
    """
    def __init__(self, n_estimators=300, max_depth=2, learning_rate=.01, M=1.35, seed=42):
        self.n_estimators=int(n_estimators); self.max_depth=int(max_depth)
        self.learning_rate=float(learning_rate); self.M=float(M); self.seed=int(seed)

    def fit(self, X, y):
        X=np.asarray(X, dtype=float); y=np.asarray(y, dtype=float).ravel()
        self.init_=float(np.median(y))
        pred=np.full_like(y, self.init_, dtype=float)
        self.trees_=[]
        for b in range(self.n_estimators):
            residual=y-pred
            # Constant factor 2 in the exact derivative is absorbed by the learning rate.
            pseudo=np.clip(residual, -self.M, self.M)
            tree=DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.seed+b)
            tree.fit(X,pseudo)
            pred += self.learning_rate*tree.predict(X)
            self.trees_.append(tree)
        return self

    def predict(self, X):
        X=np.asarray(X, dtype=float)
        out=np.full(X.shape[0], self.init_, dtype=float)
        for tree in self.trees_:
            out += self.learning_rate*tree.predict(X)
        return out
