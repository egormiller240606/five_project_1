from pathlib import Path
import json, sys, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from factor_zoo.utils.io import load_yaml
from factor_zoo.data.loaders import read_table, normalize_month_end
from factor_zoo.data.panel import build_model_panel

def main():
    root=Path(__file__).resolve().parents[1]; cfg=load_yaml(root/"configs/data.yaml")
    p=cfg["paths"]
    c=normalize_month_end(read_table(root/p["stock_characteristics"])); m=normalize_month_end(read_table(root/p["macro"])); r=normalize_month_end(read_table(root/p["monthly_returns"]))
    panel,features=build_model_panel(c,m,r,expected_industries=cfg["preprocessing"]["expected_industry_dummies"])
    out=root/p["output_panel"]; out.parent.mkdir(parents=True,exist_ok=True); panel.to_parquet(out,index=False)
    (out.parent/"feature_columns.json").write_text(json.dumps(features,indent=2),encoding="utf-8")
    print(f"saved {len(panel):,} rows, {len(features)} predictors -> {out}")
if __name__=="__main__": main()
