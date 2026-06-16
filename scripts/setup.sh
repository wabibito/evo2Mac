#!/usr/bin/env bash
#
# Set up evo2Mac on Apple Silicon.
#
# What this does:
#   1. Installs miniforge via Homebrew if missing.
#   2. Creates the `evo2Mac` conda env with Python 3.11.
#   3. Installs PyTorch (CPU/MPS wheels).
#   4. Installs `vtx` (the StripedHyena 2 runtime).
#   5. Installs this repo (`evo2` package) in editable mode without deps.
#   6. Applies patches/patch_vortex.py to the installed `vortex` package.
#
# Re-runnable: each step skips if already done.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${EVO2MAC_ENV:-evo2Mac}"
PY_VERSION="3.11"

log() { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[setup]\033[0m %s\n" "$*" >&2; }

# 1. Brew + miniforge
if ! command -v brew >/dev/null 2>&1; then
  err "Homebrew is required. Install from https://brew.sh first."
  exit 1
fi

if ! command -v conda >/dev/null 2>&1 && [[ ! -x "$HOME/miniforge3/bin/conda" ]] && [[ ! -x "/opt/homebrew/Caskroom/miniforge/base/bin/conda" ]]; then
  log "installing miniforge via Homebrew..."
  brew install --cask miniforge
else
  log "miniforge / conda already installed"
fi

# Find conda binary if not on PATH yet
if ! command -v conda >/dev/null 2>&1; then
  for cand in \
    "/opt/homebrew/Caskroom/miniforge/base/bin/conda" \
    "$HOME/miniforge3/bin/conda" \
    "/opt/miniforge3/bin/conda"; do
    if [[ -x "$cand" ]]; then
      CONDA_BIN="$cand"
      break
    fi
  done
  if [[ -z "${CONDA_BIN:-}" ]]; then
    err "could not locate conda binary after install"
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$(dirname "$CONDA_BIN")/../etc/profile.d/conda.sh"
else
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# 2. Env
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "conda env '$ENV_NAME' already exists"
else
  log "creating conda env '$ENV_NAME' (python $PY_VERSION)..."
  conda create -n "$ENV_NAME" "python=$PY_VERSION" -y
fi

conda activate "$ENV_NAME"
log "active env: $(python -V) at $(which python)"

# 3. PyTorch (MPS via the standard arm64 wheels)
if python -c "import torch" >/dev/null 2>&1; then
  log "torch already installed: $(python -c 'import torch; print(torch.__version__)')"
else
  log "installing PyTorch..."
  pip install --upgrade pip
  pip install "torch>=2.4,<3"
fi

# 4. vtx (the StripedHyena 2 runtime; imported as `vortex`)
if python -c "import vortex" >/dev/null 2>&1; then
  log "vortex already installed"
else
  log "installing vtx (StripedHyena 2 runtime)..."
  pip install vtx
fi

# 5. Editable install of our evo2 package without deps
#    (We intentionally avoid `pip install evo2` from PyPI — we want our patched
#    source in evo2/, not upstream.)
log "installing local evo2Mac package (editable, no deps)..."
pip install --no-deps -e "$REPO_ROOT"
pip install biopython huggingface_hub pyyaml "einops>=0.8" packaging rich tqdm numpy

# 6. Apply runtime vortex patches
log "applying vortex MPS patches..."
python "$REPO_ROOT/patches/patch_vortex.py"

cat <<EOF

------------------------------------------------------------
evo2Mac setup complete.

Activate the env with:

  conda activate $ENV_NAME

Smoke test (single forward pass; downloads ~4 GB from HuggingFace on first run):

  python scripts/smoke_test.py --model evo2_1b_base

Full DNA test (tokenize -> forward -> embed -> score -> generate):

  python scripts/test_dna.py --model evo2_1b_base

On a Mac with 32 GB+ unified memory you can also try:

  python scripts/test_dna.py --model evo2_7b_base

To undo the vortex patches:

  python patches/patch_vortex.py --restore
------------------------------------------------------------
EOF
