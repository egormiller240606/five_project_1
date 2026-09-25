from __future__ import annotations
import copy, numpy as np, torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def paper_huber(residual: torch.Tensor, M: float=1.35) -> torch.Tensor:
    a = residual.abs()
    return torch.where(a <= M, residual.square(), 2*M*a - M*M).mean()

class TorchHuberLinear:
    def __init__(self, M=1.35, penalty="none", lam=0.0, rho=0.5, lr=1e-2, epochs=100, batch_size=10000, patience=5, seed=42):
        self.M=M; self.penalty=penalty; self.lam=lam; self.rho=rho; self.lr=lr; self.epochs=epochs; self.batch_size=batch_size; self.patience=patience; self.seed=seed
    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.seed)
        X=np.asarray(X,np.float32); y=np.asarray(y,np.float32).reshape(-1,1)
        self.model=nn.Linear(X.shape[1],1)
        # Returns are small and the paper's zero-prediction benchmark is natural;
        # zero initialization avoids large arbitrary predictions in short debug runs.
        nn.init.zeros_(self.model.weight)
        nn.init.zeros_(self.model.bias)
        opt=torch.optim.Adam(self.model.parameters(), lr=self.lr)
        dl=DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y)),batch_size=min(self.batch_size,len(X)),shuffle=True)
        best=None; best_loss=float('inf'); wait=0
        for _ in range(self.epochs):
            self.model.train()
            for xb,yb in dl:
                opt.zero_grad(); pred=self.model(xb); loss=paper_huber(yb-pred,self.M)
                w=self.model.weight
                if self.penalty=="l1": loss=loss+self.lam*w.abs().sum()
                elif self.penalty=="elasticnet_paper": loss=loss+self.lam*((1-self.rho)*w.abs().sum()+0.5*self.rho*w.square().sum())
                loss.backward(); opt.step()
            score=self._loss(X_val,y_val) if X_val is not None else self._loss(X,y.ravel())
            if score < best_loss-1e-10: best_loss=score; best=copy.deepcopy(self.model.state_dict()); wait=0
            else:
                wait+=1
                if wait>=self.patience: break
        if best is not None: self.model.load_state_dict(best)
        return self
    def _loss(self,X,y):
        self.model.eval()
        with torch.no_grad():
            xx=torch.as_tensor(np.asarray(X,np.float32)); yy=torch.as_tensor(np.asarray(y,np.float32).reshape(-1,1))
            return float(paper_huber(yy-self.model(xx),self.M))
    def predict(self,X):
        self.model.eval()
        with torch.no_grad(): return self.model(torch.as_tensor(np.asarray(X,np.float32))).numpy().ravel()
