from pathlib import Path
import sys, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from factor_zoo.evaluation.model_averaging import average_prediction_files

root=Path(__file__).resolve().parents[1]; pred_dir=root/"outputs/predictions"
names=["pls","lasso_h","enet_h","gbrt_h","rf","vasa","nn1","nn2","nn3","nn4","nn5"]
frames={n:pd.read_parquet(pred_dir/f"{n}.parquet") for n in names if (pred_dir/f"{n}.parquet").exists()}
if len(frames)<2: raise SystemExit("Need predictions from at least two models")
out=average_prediction_files(frames); out.to_parquet(pred_dir/"model_average.parquet",index=False)
print(f"saved {len(out):,} averaged predictions from {len(frames)} models")
