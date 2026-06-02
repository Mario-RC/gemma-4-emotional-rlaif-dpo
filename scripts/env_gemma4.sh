#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GEMMA4_ROOT="${GEMMA4_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source "$GEMMA4_ROOT/vgemma4/bin/activate"

export HF_HOME="$GEMMA4_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$GEMMA4_ROOT/.cache"
export PIP_CACHE_DIR="$GEMMA4_ROOT/.cache/pip"
export TMPDIR="$GEMMA4_ROOT/tmp"
export PYTHONPATH="$GEMMA4_ROOT/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TMPDIR" "$GEMMA4_ROOT/logs"
cd "$GEMMA4_ROOT"
