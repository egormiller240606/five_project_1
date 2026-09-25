"""Generate a structured synthetic dataset that can be fed into the real pipeline."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from factor_zoo.data.synthetic import SyntheticSpec, generate_synthetic_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stocks", type=int, default=120)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--start", default="2000-01-31")
    ap.add_argument("--end", default="2020-06-30")
    ap.add_argument("--output-dir", default="data/synthetic/raw")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    spec = SyntheticSpec(n_stocks=args.n_stocks, start=args.start, end=args.end, seed=args.seed)
    meta = generate_synthetic_raw(root / args.output_dir, spec)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
