from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parents[1]; tables=root/"outputs/tables"
lines=["# Reproduction report",""]
r2=tables/"r2_oos.csv"
if r2.exists():
    d=pd.read_csv(r2); lines += ["## Out-of-sample R²","",d.to_markdown(index=False),""]
cmp=tables/"paper_comparison_table1.csv"
if cmp.exists():
    d=pd.read_csv(cmp); lines += ["## Table 1: paper vs reproduction","",d.to_markdown(index=False),""]
    if len(d): lines += [f"Mean absolute difference: **{d.abs_diff_pp.mean():.3f} percentage points**.",""]
port=tables/"portfolio_summary.csv"
if port.exists():
    d=pd.read_csv(port); lines += ["## Portfolio results","",d.to_markdown(index=False),""]
lines += ["## Interpretation note","","A numerical match may only be claimed for rows actually produced from the specified data exports. Published targets stored in the repository are reference values, not generated outputs."]
out=root/"outputs/reproduction_report.md"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text("\n".join(lines),encoding="utf-8"); print(out)
