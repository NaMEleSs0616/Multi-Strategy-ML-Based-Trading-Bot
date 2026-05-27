#!/usr/bin/env bash
# Launch the live training monitor (Streamlit).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHONPATH="$ROOT"
exec streamlit run frontend/training_monitor.py --server.headless true
