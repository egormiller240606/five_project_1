from pathlib import Path
import sys, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from factor_zoo.evaluation.subsamples import evaluate_subsamples

def main():
    root=Path(__file__).resolve().parents[1]; rows=[]
    for p in (root/"outputs/predictions").glob("*.parquet"):
        d=pd.read_parquet(p); row={"model":p.stem}; row.update(evaluate_subsamples(d)); rows.append(row)
    out=pd.DataFrame(rows).sort_values("model"); path=root/"outputs/tables/r2_oos.csv"; path.parent.mkdir(parents=True,exist_ok=True); out.to_csv(path,index=False); print(out.to_string(index=False))
if __name__=="__main__": main()
