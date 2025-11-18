#!/usr/bin/env bash
set -euo pipefail

# Always run from the repo root
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install with 'pip install uv' before running."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtual env with uv..."
  uv venv --python=3.13
fi

uv pip install -r requirements.txt --python ".venv/bin/python"

# shellcheck source=/dev/null
source ".venv/bin/activate"

exec python -m cu.agent "$@"
