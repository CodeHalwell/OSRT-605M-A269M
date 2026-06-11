#!/usr/bin/env bash
# One-shot bootstrap for finishing v6 mid-training on a Lightning AI box.
# Run this ON THE BOX after SSH is set up. It clones/pulls the repo, installs
# the env, fetches the checkpoint + snapshots the fragile gated datasets, then
# prints the exact training command.
#
# REQUIRED env vars (export them BEFORE running, or edit the block below):
#   HF_TOKEN     — HuggingFace token with the gated Nemotron grants (HallD)
#   WANDB_API_KEY— Weights & Biases key (for the osrt-v6-midtrain-lightning run)
#   MODAL_TOKEN_ID / MODAL_TOKEN_SECRET — ONLY if pulling the ckpt via Modal
#                  (alternative: scp the 3.2GB ckpt to checkpoints/v5/ yourself)
#
# Usage:
#   export HF_TOKEN=...  WANDB_API_KEY=...
#   bash scripts/lightning_bootstrap.sh
set -euo pipefail

REPO_URL="https://github.com/CodeHalwell/OSRT-605M-A269M.git"
REPO_DIR="${REPO_DIR:-OSRT-605M-A269M}"
CACHE="${HF_DATASETS_CACHE:-$PWD/hf_cache}"
CKPT="checkpoints/v5/osrt_v5_midtrain_rescue_step_3978.pt"
TOTAL_STEPS="${TOTAL_STEPS:-4500}"

echo "==> 0. preflight"
: "${HF_TOKEN:?set HF_TOKEN (gated Nemotron access) before running}"
: "${WANDB_API_KEY:?set WANDB_API_KEY before running}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || { echo "no GPU visible — aborting"; exit 1; }

echo "==> 1. clone/pull repo"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "==> 2. python env (uv preferred, else pip)"
if command -v uv >/dev/null; then
  uv sync
  PY="uv run python"
else
  python -m pip install -U pip
  python -m pip install -e . || python -m pip install torch transformers datasets wandb
  PY="python"
fi
# CUDA torch sanity (the mac/MPS wheel won't have CUDA)
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA torch required'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

echo "==> 3. checkpoint"
mkdir -p checkpoints/v5
if [ -f "$CKPT" ]; then
  echo "  ckpt present: $CKPT"
elif command -v modal >/dev/null && [ -n "${MODAL_TOKEN_ID:-}" ]; then
  echo "  pulling ckpt from Modal volume osrt-checkpoints ..."
  modal volume get osrt-checkpoints v5/osrt_v5_midtrain_rescue_step_3978.pt "$CKPT"
else
  echo "  !! ckpt missing and no Modal CLI/creds."
  echo "  !! scp it to: $REPO_DIR/$CKPT  then re-run this script."
  exit 1
fi

echo "==> 4. snapshot fragile gated datasets -> $CACHE (kills the storm)"
HF_DATASETS_CACHE="$CACHE" $PY scripts/snapshot_gated_datasets.py --cache "$CACHE"

echo ""
echo "==> 5. READY. Launch training with:"
cat <<LAUNCH

  HF_TOKEN=\$HF_TOKEN HF_HUB_OFFLINE=1 HF_DATASETS_CACHE=$CACHE \\
  WANDB_API_KEY=\$WANDB_API_KEY PYTHONPATH=src \\
  $PY scripts/lightning_midtrain.py --total-steps $TOTAL_STEPS

  # ~522 steps from the 3978 resume, cosine anneals to 2e-5 at $TOTAL_STEPS.
  # Run it under tmux/nohup so it survives an SSH drop:
  #   tmux new -s mt   (then paste the command, Ctrl-b d to detach)
LAUNCH
