import numpy as np, pandas as pd, torch
from factor_zoo.models.linear import paper_huber
from factor_zoo.models.tree import paper_huber_numpy, PaperHuberGBRT
from factor_zoo.models.vasa import marginal_r2_probabilities, VASARegressor
from factor_zoo.portfolio.sorts import portfolio_holdings, portfolio_returns
from factor_zoo.portfolio.costs import holdings_turnover, apply_turnover_costs
from factor_zoo.data.point_in_time import asof_merge_financials
from factor_zoo.features.macro import build_macro_predictors
from factor_zoo.features.catalog import STOCK_CHARACTERISTICS, FREQUENCY
from factor_zoo.evaluation.paper_targets import MONTHLY_R2, ANNUAL_R2


def test_paper_huber_exact_values():
    r=torch.tensor([0.,1.,2.])
    got=paper_huber(r,1.35).item()
    exp=np.mean([0,1,2*1.35*2-1.35**2])
    assert abs(got-exp)<1e-6
    assert np.allclose(paper_huber_numpy([1.,2.],1.35),[1.,2*1.35*2-1.35**2])


def test_vasa_sampling_prefers_predictive_feature():
    rng=np.random.default_rng(1); x=rng.normal(size=(400,3)); y=3*x[:,0]+rng.normal(scale=.1,size=400)
    p=marginal_r2_probabilities(x,y)
    assert p[0]>.95


def test_vasa_fit_predict_shape():
    rng=np.random.default_rng(2); x=rng.normal(size=(100,5)); y=x[:,0]-x[:,1]
    m=VASARegressor(10,2,1).fit(x,y)
    assert m.predict(x[:7]).shape==(7,)


def test_custom_gbrt_improves_over_constant():
    rng=np.random.default_rng(3); x=rng.normal(size=(200,2)); y=x[:,0]**2+.2*rng.normal(size=200)
    m=PaperHuberGBRT(50,2,.1,1.35,3).fit(x,y); p=m.predict(x)
    assert np.mean((y-p)**2)<np.mean((y-np.median(y))**2)


def _tiny_predictions():
    rows=[]
    for t in pd.to_datetime(["2020-01-31","2020-02-29"]):
        for i in range(20):
            rows.append({"date":t,"stock_id":f"S{i}","prediction":i,"target":i/1000,"future_return":i/1000,"market_cap":i+1})
    return pd.DataFrame(rows)


def test_portfolio_weights_have_correct_exposure():
    h=portfolio_holdings(_tiny_predictions())
    for _,g in h.groupby("date"):
        assert np.isclose(g.w_long_only.sum(),1)
        assert np.isclose(g.w_long_short[g.w_long_short>0].sum(),1)
        assert np.isclose(g.w_long_short[g.w_long_short<0].sum(),-1)


def test_portfolio_return_identity():
    d=_tiny_predictions(); h=portfolio_holdings(d); r=portfolio_returns(d)
    first=h[h.date==h.date.min()]
    assert np.isclose(r.iloc[0].long_only,(first.w_long_only*first.target).sum())


def test_turnover_nonnegative_and_cost_monotone():
    d=_tiny_predictions(); h=portfolio_holdings(d); r=portfolio_returns(d)
    t=holdings_turnover(h,"w_long_only"); assert (t>=0).all()
    z20=apply_turnover_costs(r,h,20); z80=apply_turnover_costs(r,h,80)
    assert (z80.long_only_net_80bps<=z20.long_only_net_20bps+1e-15).all()


def test_point_in_time_never_uses_future_statement():
    monthly=pd.DataFrame({"stock_id":["A","A"],"date":pd.to_datetime(["2020-03-31","2020-04-30"])})
    st=pd.DataFrame({"stock_id":["A"],"available_date":pd.to_datetime(["2020-04-15"]),"assets":[10]})
    out=asof_merge_financials(monthly,st)
    assert pd.isna(out.iloc[0].assets) and out.iloc[1].assets==10


def test_macro_term_spread_and_growth():
    dates=pd.date_range("2019-01-31",periods=13,freq="ME")
    d=pd.DataFrame({"date":dates,"gov10y_yield":3.,"gov1y_yield":2.,"m2":np.arange(1,14.),"trade_volume":np.arange(2,15.),"market_turnover":.5})
    o=build_macro_predictors(d)
    assert np.allclose(o.tms,1) and "m2gr" in o and "itgr" in o and "mtr" in o


def test_catalog_counts_match_paper():
    assert len(STOCK_CHARACTERISTICS)==90
    vals=[FREQUENCY[x] for x in STOCK_CHARACTERISTICS]
    # Paper count 94 includes four ownership dummies, all annual.
    assert vals.count("monthly")==22
    assert vals.count("semiannual")==6
    assert vals.count("annual")==11
    assert vals.count("quarterly")==51


def test_paper_target_tables_complete():
    assert len(MONTHLY_R2)==7 and all(len(v)==13 for v in MONTHLY_R2.values())
    assert len(ANNUAL_R2)==7 and all(len(v)==13 for v in ANNUAL_R2.values())
    assert MONTHLY_R2["all"][5]==2.71


def test_performance_loss_metrics_positive_magnitude():
    from factor_zoo.portfolio.performance import stats
    s=stats([.1,-.2,.05])
    assert s["max_drawdown"]>=0 and s["max_1m_loss"]==.2
