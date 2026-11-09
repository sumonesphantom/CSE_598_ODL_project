"""ImageNet-C style corruption error (CE / mCE) between two diagnostic runs.

Typical use: compare backbones, each run with scripts/run_diagnostic.py --backbone ... --out ...

  python scripts/benchmark.py --model results/diagnostic_vit --reference results/diagnostic

Outputs (in --out):
  ce.csv      per perturbation: mean error of model and reference, CE, relative CE
  mce.json    mCE and relative mCE (held-out perturbations excluded)
"""

import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from bolero.benchmark import corruption_errors, error_table, mean_corruption_error
from bolero.config import RESULTS_DIR
from bolero.perturbations import HELD_OUT
from bolero.records import save


def load_run(path: Path) -> tuple[pd.DataFrame, float]:
    summary = pd.read_csv(path / "summary.csv")
    clean_acc = json.loads((path / "headline.json").read_text())["clean_accuracy"]
    return error_table(summary), clean_acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="diagnostic output dir of the model")
    parser.add_argument("--reference", type=Path, default=RESULTS_DIR / "diagnostic")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "benchmark")
    args = parser.parse_args()

    model, model_clean = load_run(args.model)
    ref, ref_clean = load_run(args.reference)
    ce = corruption_errors(model, ref, model_clean, ref_clean)
    save(ce, args.out / "ce.csv")
    headline = {"model": str(args.model), "reference": str(args.reference),
                "clean_error_model": 1.0 - model_clean, "clean_error_ref": 1.0 - ref_clean,
                **mean_corruption_error(ce, exclude=tuple(HELD_OUT))}
    (args.out / "mce.json").write_text(json.dumps(headline, indent=2))
    print(ce.to_string(index=False))
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
