from pathlib import Path
import sys, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from factor_zoo.evaluation.compare import compare_monthly_r2

root=Path(__file__).resolve().parents[1]
p=root/"outputs/tables/r2_oos.csv"
if not p.exists(): raise SystemExit("Run scripts/03_evaluate_models.py first")
out=compare_monthly_r2(pd.read_csv(p))
out.to_csv(root/"outputs/tables/paper_comparison_table1.csv",index=False)
print(out.to_string(index=False))
if len(out): print(f"\nMean absolute difference: {out.abs_diff_pp.mean():.3f} percentage points")
