from pathlib import Path
import sys, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from factor_zoo.portfolio.sorts import portfolio_returns, portfolio_holdings
from factor_zoo.portfolio.costs import apply_turnover_costs
from factor_zoo.portfolio.performance import stats


def main():
    root=Path(__file__).resolve().parents[1]; summary=[]
    (root/"outputs/tables").mkdir(parents=True,exist_ok=True)
    for p in (root/"outputs/predictions").glob("*.parquet"):
        d=pd.read_parquet(p)
        if "market_cap" not in d: continue
        if "future_return" not in d: d["future_return"]=d["target"]
        h=portfolio_holdings(d); pr=portfolio_returns(d)
        h.to_parquet(root/"outputs"/f"holdings_{p.stem}.parquet",index=False)
        pr.to_csv(root/"outputs/tables"/f"portfolio_{p.stem}.csv",index=False)
        for bps in [0,20,40,60,80]:
            z=apply_turnover_costs(pr,h,bps)
            a=stats(z[f"long_only_net_{bps}bps"]); b=stats(z[f"long_short_net_{bps}bps"])
            summary += [{"model":p.stem,"strategy":"long_only","cost_bps":bps,**a},{"model":p.stem,"strategy":"long_short","cost_bps":bps,**b}]
    pd.DataFrame(summary).to_csv(root/"outputs/tables/portfolio_summary.csv",index=False)
if __name__=="__main__": main()
