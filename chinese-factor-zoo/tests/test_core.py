import numpy as np, pandas as pd
from factor_zoo.evaluation.metrics import r2_oos
from factor_zoo.training.splits import expanding_yearly_splits
from factor_zoo.features.preprocessing import cross_sectional_rank_to_unit_interval

def test_r2_oos_perfect(): assert r2_oos([1,2],[1,2])==1.0

def test_first_split():
    s=next(expanding_yearly_splits())
    assert str(s.train_end.date())=="2008-12-31" and str(s.val_start.date())=="2009-01-31" and str(s.test_start.date())=="2012-01-31"

def test_rank_bounds():
    d=pd.DataFrame({"date":["2020-01-31"]*3,"x":[1.,2.,3.]}); o=cross_sectional_rank_to_unit_interval(d,["x"])
    assert o.x.min()>=-1 and o.x.max()<=1

def test_feature_count_with_80_industries():
    from factor_zoo.features.catalog import STOCK_CHARACTERISTICS, MACRO_VARIABLES
    from factor_zoo.data.panel import build_model_panel
    rng=np.random.default_rng(2); dates=pd.date_range("2010-01-31",periods=2,freq="ME"); stocks=[f"S{i:02d}" for i in range(80)]
    idx=pd.MultiIndex.from_product([stocks,dates],names=["stock_id","date"]).to_frame(index=False); c=idx.copy()
    for x in STOCK_CHARACTERISTICS: c[x]=rng.normal(size=len(c))
    c["industry"]=[f"I{i:02d}" for i in range(80) for _ in range(2)]; c["market_cap"]=1.; c["soe"]=0
    m=pd.DataFrame({"date":dates});
    for x in MACRO_VARIABLES: m[x]=rng.normal(size=len(m))
    r=idx.copy(); r["return"]=rng.normal(size=len(r)); r["risk_free"]=0.
    _,features=build_model_panel(c,m,r,expected_industries=80)
    assert len(features)==1160
