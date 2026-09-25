from __future__ import annotations
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from factor_zoo.models.linear import TorchHuberLinear
from factor_zoo.models.neural import NeuralRegressor
from factor_zoo.models.vasa import VASARegressor
from factor_zoo.models.tree import PaperHuberGBRT

def build_model(name,cfg,params=None,seed=42,M=1.35):
    params=params or {}; c=cfg["models"][name]; fam=c["family"]
    if fam=="torch_linear":
        return TorchHuberLinear(M=M, penalty=c.get("penalty","none"), lam=params.get("lambda",0), rho=c.get("rho",.5), lr=c.get("lr",.01), epochs=c.get("epochs",100), batch_size=c.get("batch_size",10000), patience=c.get("patience",5), seed=seed)
    if fam=="pls": return PLSRegression(n_components=params.get("n_components",5), scale=False)
    if fam=="paper_gbrt": return PaperHuberGBRT(max_depth=params.get("max_depth",2), n_estimators=params.get("max_iter",300), learning_rate=params.get("learning_rate",.01), M=M, seed=seed)
    if fam=="random_forest": return RandomForestRegressor(max_depth=params.get("max_depth",5), n_estimators=params.get("n_estimators",200), max_features=params.get("max_features",20), n_jobs=c.get("n_jobs",-1), random_state=seed)
    if fam=="vasa": return VASARegressor(params.get("n_submodels",100),params.get("n_components",20),seed)
    if fam=="neural_net":
        d=cfg["neural_defaults"]; return NeuralRegressor(c["hidden_layers"],M=M,l1=params.get("l1",1e-4),lr=params.get("lr",1e-3),batch_size=params.get("batch_size",2048),epochs=d["epochs"],patience=d["patience"],dropout=params.get("dropout",d.get("dropout",0.2)),batch_norm=d["batch_norm"],seed=seed)
    raise KeyError(name)
