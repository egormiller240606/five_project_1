from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt

def main():
    root=Path(__file__).resolve().parents[1]; od=root/"outputs/figures"; od.mkdir(parents=True,exist_ok=True)
    for p in (root/"outputs/tables").glob("portfolio_*.csv"):
        d=pd.read_csv(p,parse_dates=["date"])
        for strategy in ["long_only","long_short"]:
            fig,ax=plt.subplots(figsize=(9,5)); ax.plot(d.date,(1+d[strategy]).cumprod()); ax.set(title=f"{p.stem}: {strategy}",xlabel="Date",ylabel="Growth of 1"); fig.tight_layout(); fig.savefig(od/f"{p.stem}_{strategy}.png",dpi=160); plt.close(fig)
if __name__=="__main__": main()
