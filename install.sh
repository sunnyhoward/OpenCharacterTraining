#!/usr/bin/env bash
# =============================================================================
# OpenCharacterTraining one-step installer (idempotent — safe to re-run).
#
#   Usage:  bash install.sh
#
# Sets up everything the pipeline needs that a plain `git clone` does NOT give
# you: the OpenRLHF submodule working tree, editable installs of this repo and
# the OpenRLHF fork, the python deps (incl. vllm/torchdata/optree), and a local
# character/constants.py. run_pipeline.sh calls this automatically on first run.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VENV="${VENV:-/venv/main}"
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"

# Prefer uv (fast) when available, else plain pip.
PIP="pip"
command -v uv >/dev/null 2>&1 && PIP="uv pip"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# -----------------------------------------------------------------------------
log "[1/5] submodules (openrlhf required; repeng optional)"
# repeng's recorded URL is SSH (git@github.com); rewrite to HTTPS so it clones
# without an SSH key. Harmless if you already use SSH.
git config url."https://github.com/".insteadOf git@github.com: 2>/dev/null || true
git submodule update --init openrlhf 2>/dev/null || true
git submodule update --init repeng   2>/dev/null || echo "   repeng unavailable — skipping (only needed for steering/preferences eval)"
# A submodule can be 'initialized' yet have an empty working tree (an aborted
# recursive clone). Force the checkout if the package files are missing.
if [[ ! -f openrlhf/setup.py ]]; then
  log "   openrlhf working tree empty — forcing checkout"
  git -C openrlhf checkout -f HEAD
fi

# -----------------------------------------------------------------------------
log "[2/5] python dependencies (requirements.txt: vllm, torchdata, optree, ...)"
# Note: this installs vllm, which pins torch==2.11.x. On a fresh box that may
# replace a newer preinstalled torch — expected and required by vllm.
$PIP install -r requirements.txt

# -----------------------------------------------------------------------------
log "[3/5] OpenRLHF fork (editable)"
# --no-deps on purpose: OpenRLHF pins older transformers/deepspeed that would
# downgrade (and break) the versions vllm + the data scripts need. Everything
# OpenRLHF actually imports is already provided by requirements.txt above.
$PIP install -e openrlhf --no-deps

# -----------------------------------------------------------------------------
log "[4/5] OpenCharacterTraining package (editable)"
$PIP install -e . --no-deps

# -----------------------------------------------------------------------------
log "[5/5] character/constants.py (local path config; gitignored)"
if [[ -f character/constants.py ]]; then
  echo "   exists — leaving as-is"
else
  cp character/constants.py.example character/constants.py
  echo "   created from character/constants.py.example"
fi

# -----------------------------------------------------------------------------
log "verifying imports"
python - <<'PY'
import importlib
for m in ["character.constants", "openrlhf.cli.train_dpo", "vllm", "torchdata", "optree"]:
    importlib.import_module(m)
    print("  OK", m)
PY

touch "$HERE/.install_complete"
log "environment ready."
