#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Create with: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

source .venv/bin/activate

echo "Running EDA..."
MPLBACKEND=Agg python -m src.eda.run_eda

echo "Running transform pipeline..."
python -m src.processing.pipeline

echo "Running splits..."
python -m src.models.splits

echo "Running baseline models..."
python -m src.models.baseline

echo "Running elasticity and price curves..."
python -m src.models.elasticity

echo "Running evaluation..."
python -m src.models.eval

echo "Done. Reports and processed data updated."
