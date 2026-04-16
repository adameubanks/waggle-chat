#!/bin/bash
#SBATCH --job-name=waggle-train
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --output=runs/slurm/%x-%j.out
#SBATCH --error=runs/slurm/%x-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

module load cuda

cd /nobackup/archive/usr/adameuba/waggle_chat
mkdir -p runs/slurm

ENV_PREFIX="${ENV_PREFIX:-/home/adameuba/.conda/envs/waggle_env}"
RUN_DIR="runs/train/${SLURM_JOB_ID:-local}"
MAN_DIR="${RUN_DIR}/manifests"
CACHE_DIR="${RUN_DIR}/cache"
mkdir -p "$RUN_DIR" "$MAN_DIR"

echo "job=${SLURM_JOB_ID:-?} node=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "RUN_DIR=$RUN_DIR"

mamba run -p "$ENV_PREFIX" python - <<'PY'
import sys
try:
    import cv2  # noqa: F401
except Exception as e:
    print(f"FATAL: cv2 import failed: {e}", file=sys.stderr)
    raise
print("cv2 import ok")
PY

mamba run -p "$ENV_PREFIX" python run.py prep \
  --data_dir data \
  --out_dir "$MAN_DIR" \
  --fps 30 \
  --clip_frames 16 \
  --stride 2 \
  --center_stride_frames 2 \
  --neg_per_pos 3 \
  --seed 0

mamba run -p "$ENV_PREFIX" python run.py cache \
  --manifest_csv "$MAN_DIR/train_manifest.csv" \
  --out_dir "$CACHE_DIR/train" \
  --fps 30 \
  --clip_frames 16 \
  --stride 2 \
  --shard_size 512

mamba run -p "$ENV_PREFIX" python run.py cache \
  --manifest_csv "$MAN_DIR/val_manifest.csv" \
  --out_dir "$CACHE_DIR/val" \
  --fps 30 \
  --clip_frames 16 \
  --stride 2 \
  --shard_size 512

mamba run -p "$ENV_PREFIX" python run.py train \
  --train_manifest "$MAN_DIR/train_manifest.csv" \
  --val_manifest "$MAN_DIR/val_manifest.csv" \
  --out_dir "$RUN_DIR" \
  --cache_train_dir "$CACHE_DIR/train" \
  --cache_val_dir "$CACHE_DIR/val" \
  --epochs 20 \
  --batch_size 64 \
  --lr 3e-4 \
  --weight_decay 5e-3 \
  --num_workers 2 \
  --prefetch_factor 2 \
  --val_every 1 \
  --val_max_batches 60 \
  --amp

ls -lah "$RUN_DIR"/*.pt || true

