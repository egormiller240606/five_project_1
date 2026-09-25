"""Canonical raw-to-characteristic helpers.

CSMAR/WIND field codes differ by subscription/export. Map exported columns to the canonical
names below, then these functions reproduce the appendix formulas that are identifiable from the
paper. Complex scores (Piotroski/Mohanram), ATR and er_trend are intentionally separate because
additional original-paper definitions/event dummies are required.
"""
from __future__ import annotations
import numpy as np, pandas as pd

def safe_div(a,b):
    b=np.asarray(b,float); return np.asarray(a,float)/np.where(np.abs(b)<1e-12,np.nan,b)

def pct_change(s,periods=1): return s.groupby(level=0).pct_change(periods)

def accounting_characteristics(df: pd.DataFrame, stock_col="stock_id", date_col="date", industry_col="industry") -> pd.DataFrame:
    d=df.sort_values([stock_col,date_col]).copy(); g=d.groupby(stock_col,group_keys=False)
    lag=lambda c,n=1:g[c].shift(n)
    ch=lambda c,n=1:d[c]-lag(c,n)
    avg=lambda c:(d[c]+lag(c))/2
    # Canonical required fields are named transparently; missing formulas are skipped.
    def has(*c): return all(x in d for x in c)
    if has("current_assets","cash","current_liabilities","short_term_debt","tax_payable","depreciation","total_assets"):
        num=(ch("current_assets")-ch("cash"))-(ch("current_liabilities")-ch("short_term_debt")-ch("tax_payable"))-d["depreciation"]
        d["acc"]=safe_div(num,d["total_assets"]); d["absacc"]=d["acc"].abs()
    if has("total_assets"): d["agr"]=g["total_assets"].pct_change(4)
    if has("book_equity","market_cap"): d["bm"]=safe_div(d.book_equity,d.market_cap)
    if has("cash","total_assets"): d["cash"]=safe_div(d.cash,avg("total_assets"))
    if has("earnings","total_liabilities"): d["cashdebt"]=safe_div(d.earnings,d.total_liabilities)
    if has("market_cap","long_term_debt","total_assets","cash"): d["cashspr"]=safe_div(d.market_cap+d.long_term_debt-d.total_assets,d.cash)
    if has("operating_cash_flow","market_cap"): d["cfp"]=safe_div(d.operating_cash_flow,d.market_cap)
    if has("sales","total_assets"): d["chato"]=safe_div(ch("sales"),avg("total_assets"))
    if has("shares_outstanding"): d["chcsho"]=g["shares_outstanding"].pct_change()
    if has("inventory","total_assets"): d["chinv"]=safe_div(ch("inventory"),d.total_assets)
    if has("income_before_extraordinary","sales"): d["chpm"]=safe_div(ch("income_before_extraordinary"),d.sales)
    if has("tax_expense"): d["chtx"]=g["tax_expense"].pct_change()
    if has("current_assets","current_liabilities"): d["currat"]=safe_div(d.current_assets,d.current_liabilities)
    if has("depreciation","fixed_assets"): d["depr"]=safe_div(d.depreciation,d.fixed_assets)
    if has("book_equity"): d["egr"]=g["book_equity"].pct_change()
    if has("revenue","cogs","total_assets"): d["gma"]=safe_div(d.revenue-d.cogs,lag("total_assets"))
    if has("employees"): d["hire"]=g["employees"].pct_change(4)
    if has("fixed_assets","inventory","total_assets"): d["invest"]=safe_div(ch("fixed_assets",4)+ch("inventory",4),lag("total_assets",4))
    if has("total_liabilities","market_cap"): d["lev"]=safe_div(d.total_liabilities,d.market_cap)
    if has("total_liabilities"): d["lgr"]=g["total_liabilities"].pct_change()
    if has("market_cap"): d["mve"]=np.log(d.market_cap.where(d.market_cap>0))
    if has("operating_profit","book_equity"): d["operprof"]=safe_div(d.operating_profit,lag("book_equity"))
    if has("current_assets","inventory","current_liabilities"): d["quick"]=safe_div(d.current_assets-d.inventory,d.current_liabilities)
    if has("rd_expense","total_assets"): d["rd"]=(safe_div(d.rd_expense,d.total_assets)-safe_div(lag("rd_expense"),lag("total_assets"))>.05).astype(float)
    if has("rd_expense","market_cap"): d["rd_mve"]=safe_div(d.rd_expense,d.market_cap)
    if has("rd_expense","sales"): d["rd_sale"]=safe_div(d.rd_expense,d.sales)
    if has("investment_real_estate","fixed_assets"): d["realestate"]=safe_div(d.investment_real_estate,d.fixed_assets)
    if has("income_before_extraordinary","total_assets"): d["roaq"]=safe_div(d.income_before_extraordinary,lag("total_assets"))
    if has("income_before_extraordinary","book_equity"): d["roeq"]=safe_div(d.income_before_extraordinary,lag("book_equity"))
    if has("sales","market_cap"): d["rsup"]=safe_div(ch("sales"),d.market_cap); d["sp"]=safe_div(d.sales,d.market_cap)
    if has("sales","cash"): d["salecash"]=safe_div(d.sales,d.cash)
    if has("sales","inventory"): d["saleinv"]=safe_div(d.sales,d.inventory)
    if has("sales","receivables"): d["salerev"]=safe_div(d.sales,d.receivables)
    if has("sales"): d["sgr"]=g["sales"].pct_change()
    if has("cash","receivables","inventory","fixed_assets","total_assets"):
        d["tang"]=safe_div(d.cash+.715*d.receivables+.547*d.inventory+.535*d.fixed_assets,d.total_assets)
    if has("current_tax_expense","total_income"): d["tb"]=safe_div(d.current_tax_expense/.25,d.total_income)
    # Industry adjustments in the paper are equal-weighted within industry.
    for src,dst in [("bm","bm_ia"),("cfp","cfp_ia"),("chato","chatoia"),("chpm","chpmia"),("mve","mve_ia")]:
        if src in d and industry_col in d: d[dst]=d[src]-d.groupby([date_col,industry_col])[src].transform("mean")
    return d

def monthly_market_characteristics(monthly: pd.DataFrame, stock_col="stock_id", date_col="date") -> pd.DataFrame:
    d=monthly.sort_values([stock_col,date_col]).copy(); g=d.groupby(stock_col,group_keys=False)
    if "return" in d:
        d["mom1m"]=g["return"].shift(1)
        for name,lo,hi in [("mom6m",2,6),("mom12m",2,12),("mom36m",13,36)]:
            d[name]=g["return"].transform(lambda s:(1+s.shift(lo-1)).rolling(hi-lo+1).apply(np.prod,raw=True)-1)
        d["chmom"]=g["return"].transform(lambda s:((1+s.shift(1)).rolling(6).apply(np.prod,raw=True)-1)-((1+s.shift(7)).rolling(6).apply(np.prod,raw=True)-1))
    if "market_cap" in d: d["mve"]=np.log(g["market_cap"].shift(1).where(g["market_cap"].shift(1)>0))
    if "shares_outstanding" in d: d["chcsho"]=g["shares_outstanding"].pct_change()
    return d

def daily_to_monthly(daily: pd.DataFrame,stock_col="stock_id",date_col="date") -> pd.DataFrame:
    d=daily.copy(); d[date_col]=pd.to_datetime(d[date_col]); d["month"]=d[date_col].dt.to_period("M").dt.to_timestamp("M")
    rec=[]
    for (sid,m),g in d.groupby([stock_col,"month"]):
        z={stock_col:sid,"date":m}
        if "return" in g: z["maxret"]=g["return"].max(); z["volatility"]=g["return"].std(ddof=1)
        if "rmb_volume" in g:
            z["std_dolvol"]=g["rmb_volume"].std(ddof=1); z["ill"]=np.nanmean(np.abs(g.get("return",0))/g["rmb_volume"].replace(0,np.nan))
        if "turnover" in g: z["std_turn"]=g["turnover"].std(ddof=1)
        rec.append(z)
    return pd.DataFrame(rec)
