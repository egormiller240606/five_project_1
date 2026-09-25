from __future__ import annotations
import copy, numpy as np, torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from factor_zoo.models.linear import paper_huber

class FeedForwardNet(nn.Module):
    def __init__(self,p,hidden,dropout=.2,batch_norm=True):
        super().__init__(); layers=[]; d=p
        for h in hidden:
            layers.append(nn.Linear(d,h));
            if batch_norm: layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout>0: layers.append(nn.Dropout(dropout))
            d=h
        final = nn.Linear(d, 1)
        # Monthly returns are centered near zero; start from the paper's natural
        # zero-forecast benchmark instead of an arbitrary random return level.
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class NeuralRegressor:
    def __init__(self,hidden_layers,M=1.35,l1=0.0,lr=1e-3,batch_size=2048,epochs=100,patience=5,dropout=.2,batch_norm=True,seed=42):
        self.hidden_layers=hidden_layers; self.M=M; self.l1=l1; self.lr=lr; self.batch_size=batch_size; self.epochs=epochs; self.patience=patience; self.dropout=dropout; self.batch_norm=batch_norm; self.seed=seed
    def fit(self,X,y,X_val,y_val):
        torch.manual_seed(self.seed); X=np.asarray(X,np.float32); y=np.asarray(y,np.float32).reshape(-1,1)
        self.model=FeedForwardNet(X.shape[1],self.hidden_layers,self.dropout,self.batch_norm)
        opt=torch.optim.Adam(self.model.parameters(),lr=self.lr)
        dl=DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y)),batch_size=min(self.batch_size,len(X)),shuffle=True)
        best=None; best_loss=float('inf'); wait=0
        for _ in range(self.epochs):
            self.model.train()
            for xb,yb in dl:
                opt.zero_grad(); pred=self.model(xb); loss=paper_huber(yb-pred,self.M)
                if self.l1: loss=loss+self.l1*sum(p.abs().sum() for n,p in self.model.named_parameters() if "weight" in n)
                loss.backward(); opt.step()
            score=self._mse(X_val,y_val)
            if score<best_loss-1e-10: best_loss=score; best=copy.deepcopy(self.model.state_dict()); wait=0
            else:
                wait+=1
                if wait>=self.patience: break
        self.model.load_state_dict(best); return self
    def _mse(self,X,y):
        p=self.predict(X); return float(np.mean((np.asarray(y)-p)**2))
    def predict(self,X):
        self.model.eval()
        with torch.no_grad(): return self.model(torch.as_tensor(np.asarray(X,np.float32))).numpy().ravel()
