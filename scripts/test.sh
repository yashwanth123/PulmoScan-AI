#!/usr/bin/env bash
# Run the full PulmoScan AI test suite
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Installing test dependencies"
pip install -q -r requirements-dev.txt

echo "==> Running pytest"
python3 -m pytest tests/ -v --tb=short "$@"

echo "==> All tests passed"
