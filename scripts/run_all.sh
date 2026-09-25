#!/usr/bin/env bash
# Run the full pipeline. Usage:
#   scripts/run_all.sh quick   # ~10 min on an M-series Mac: 5 images/class
#   scripts/run_all.sh full    # all 3,925 val images for the diagnostic, 50/class for restoration
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-quick}"
if [[ "$MODE" == "quick" ]]; then
  DIAG="--per-class 5"; TRAIN="--train-per-class 10 --val-per-class 5"; REST="--per-class 5"; SWEEP="--per-class 5"
elif [[ "$MODE" == "full" ]]; then
  DIAG=""; TRAIN="--train-per-class 40 --val-per-class 20"; REST="--per-class 50"; SWEEP="--per-class 50"
else
  echo "usage: $0 [quick|full]"; exit 1
fi

python scripts/prepare_data.py
python scripts/run_diagnostic.py $DIAG
python scripts/train_detector.py $TRAIN
python scripts/run_restoration.py $REST
python scripts/run_cue_sweep.py $SWEEP
python scripts/analyze.py
