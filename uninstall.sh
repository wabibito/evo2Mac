#!/usr/bin/env bash
#
# Reverse what install.sh did. Everything evo2Mac touches is contained in:
#
#   1. The `evo2Mac` conda env (created by install.sh).
#   2. The .bak files inside the installed `vortex` package (created by
#      patches/patch_vortex.py).
#   3. The HuggingFace cache directory (~/.cache/huggingface/) — only the
#      Evo 2 checkpoints, not other models you may have downloaded.
#
# This script is interactive — it asks before removing each piece.
#
# Use --yes to skip prompts (CI / scripted).
#
# Not touched: Homebrew, miniforge itself, your other conda envs.

set -euo pipefail

ENV_NAME="${EVO2MAC_ENV:-evo2Mac}"
HF_CACHE_DEFAULT="$HOME/.cache/huggingface"
HF_CACHE="${HF_HOME:-$HF_CACHE_DEFAULT}"

ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    -y|--yes) ASSUME_YES=1 ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0 ;;
  esac
done

log() { printf "\033[1;34m[uninstall]\033[0m %s\n" "$*"; }
ask() {
  if [[ "$ASSUME_YES" -eq 1 ]]; then
    return 0
  fi
  read -r -p "$1 [y/N] " ans
  [[ "${ans:-}" =~ ^[Yy]$ ]]
}

if ! command -v conda >/dev/null 2>&1; then
  for cand in \
    "/opt/homebrew/Caskroom/miniforge/base/bin/conda" \
    "$HOME/miniforge3/bin/conda"; do
    if [[ -x "$cand" ]]; then
      # shellcheck disable=SC1091
      source "$(dirname "$cand")/../etc/profile.d/conda.sh"
      break
    fi
  done
fi

# 1. Revert the runtime patches to vortex (only matters if you ever want to
#    keep the env but uninstall just evo2Mac — usually you'd nuke the env).
if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "$REPO_ROOT/patches/patch_vortex.py" ]]; then
    if ask "Restore the vortex package from .bak files (undo the Mac patches)?"; then
      conda run -n "$ENV_NAME" python "$REPO_ROOT/patches/patch_vortex.py" --restore || true
    fi
  fi
fi

# 2. Remove the conda env.
if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if ask "Remove the '$ENV_NAME' conda env (frees ~3-4 GB)?"; then
    conda env remove -n "$ENV_NAME" -y
    log "removed env '$ENV_NAME'"
  fi
else
  log "no '$ENV_NAME' env found — nothing to remove"
fi

# 3. Remove the HuggingFace cache for Evo 2 checkpoints.
EVO2_DIRS=()
for d in \
  "$HF_CACHE/hub/models--arcinstitute--evo2_1b_base" \
  "$HF_CACHE/hub/models--arcinstitute--evo2_7b" \
  "$HF_CACHE/hub/models--arcinstitute--evo2_7b_base" \
  "$HF_CACHE/hub/models--arcinstitute--evo2_7b_262k" \
  "$HF_CACHE/hub/models--arcinstitute--evo2_7b_microviridae" \
  "$HF_CACHE/hub/models--arcinstitute--evo2_20b" \
  "$HF_CACHE/hub/models--arcinstitute--evo2_40b" \
  "$HF_CACHE/hub/models--arcinstitute--evo2_40b_base"; do
  if [[ -d "$d" ]]; then
    EVO2_DIRS+=("$d")
  fi
done

if [[ ${#EVO2_DIRS[@]} -gt 0 ]]; then
  log "found Evo 2 checkpoints in HF cache:"
  for d in "${EVO2_DIRS[@]}"; do
    size=$(du -sh "$d" 2>/dev/null | awk '{print $1}')
    printf "    %s  (%s)\n" "$d" "$size"
  done
  if ask "Delete these checkpoint directories?"; then
    for d in "${EVO2_DIRS[@]}"; do
      rm -rf "$d"
      log "removed $d"
    done
  fi
else
  log "no Evo 2 checkpoints found in HF cache"
fi

# 4. Also remove merged .pt files (evo2 wrapper writes these next to ~/.cache/huggingface/hub)
MERGED_PT_DIR="$(dirname "$HF_CACHE/hub")"
PT_FILES=()
for f in "$MERGED_PT_DIR"/evo2_*.pt; do
  [[ -f "$f" ]] && PT_FILES+=("$f")
done
if [[ ${#PT_FILES[@]} -gt 0 ]]; then
  log "found merged checkpoint files:"
  for f in "${PT_FILES[@]}"; do
    size=$(du -sh "$f" 2>/dev/null | awk '{print $1}')
    printf "    %s  (%s)\n" "$f" "$size"
  done
  if ask "Delete these merged .pt files?"; then
    rm -f "${PT_FILES[@]}"
    log "removed merged .pt files"
  fi
fi

cat <<EOF

------------------------------------------------------------
Uninstall complete. NOT touched:
  - Homebrew
  - miniforge itself
  - your other conda envs
  - the cloned evo2Mac repo (delete with: rm -rf "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")
------------------------------------------------------------
EOF
