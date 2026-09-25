from pathlib import Path
import argparse, json, sys, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from factor_zoo.utils.io import load_yaml
from factor_zoo.training.runner import run_expanding_experiment

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--models",nargs="+",default=["ols3_h","lasso_h","enet_h","rf","gbrt_h","nn4"]); ap.add_argument("--max-candidates",type=int,default=None); a=ap.parse_args()
    root=Path(__file__).resolve().parents[1]; panel=pd.read_parquet(root/"data/processed/model_panel.parquet"); features=json.loads((root/"data/processed/feature_columns.json").read_text())
    mc=load_yaml(root/"configs/models.yaml"); ec=load_yaml(root/"configs/experiment.yaml")
    outdir=root/"outputs/predictions"; outdir.mkdir(parents=True,exist_ok=True)
    for name in a.models:
        pred,chosen=run_expanding_experiment(panel,features,name,mc,ec,max_candidates=a.max_candidates)
        pred.to_parquet(outdir/f"{name}.parquet",index=False); (outdir/f"{name}_params.json").write_text(json.dumps(chosen,indent=2),encoding="utf-8")
        print(name,len(pred))
if __name__=="__main__": main()
