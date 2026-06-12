#!/usr/bin/env bash
# Create a local venv and install galacticsics with dev dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! python3 -m venv .venv 2>/dev/null; then
  echo "python3-venv is required. On Debian/Ubuntu: sudo apt install python3-venv"
  exit 1
fi

.venv/bin/pip install -U pip
.venv/bin/pip install -e ".[dev]"
echo "Done. Activate with: source .venv/bin/activate"
echo "Run tests: pytest tests/ -v"
