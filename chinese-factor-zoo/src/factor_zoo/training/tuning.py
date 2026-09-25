from __future__ import annotations
import itertools, numpy as np
from factor_zoo.evaluation.metrics import mse
from factor_zoo.models.registry import build_model


def candidate_params(name, cfg):
    c=cfg["models"][name]; fam=c["family"]
    if fam=="pls": return [{"n_components":k} for k in c["n_components_grid"]]
    if fam=="torch_linear" and "lambda_grid" in c: return [{"lambda":x} for x in c["lambda_grid"]]
    if fam=="paper_gbrt": return [dict(zip(("max_depth","max_iter","learning_rate"),v)) for v in itertools.product(c["max_depth_grid"],c["max_iter_grid"],c["learning_rate_grid"])]
    if fam=="random_forest": return [dict(zip(("max_depth","n_estimators","max_features"),v)) for v in itertools.product(c["max_depth_grid"],c["n_estimators_grid"],c["max_features_grid"])]
    if fam=="vasa": return [dict(zip(("n_submodels","n_components"),v)) for v in itertools.product(c["n_submodels_grid"],c["n_components_grid"])]
    if fam=="neural_net":
        d=cfg["neural_defaults"]
        return [dict(zip(("l1","lr","batch_size","dropout"),v)) for v in itertools.product(d["l1_grid"],d["lr_grid"],d["batch_size_grid"],d.get("dropout_grid",[d.get("dropout",0.2)]))]
    return [{}]


def fit_any(model, Xtr,ytr,Xv,yv):
    try: return model.fit(Xtr,ytr,Xv,yv)
    except TypeError: return model.fit(Xtr,ytr)


def tune_model(name,cfg,Xtr,ytr,Xv,yv,max_candidates=None):
    best=None
    cand=candidate_params(name,cfg)
    if max_candidates: cand=cand[:max_candidates]
    for params in cand:
        m=build_model(name,cfg,params,seed=cfg.get("seed",42),M=cfg.get("huber_M",1.35))
        fit_any(m,Xtr,ytr,Xv,yv); pred=np.asarray(m.predict(Xv)).ravel(); score=mse(yv,pred)
        if best is None or score<best[0]: best=(score,params,m)
    return best
