#!/bin/bash
#SBATCH --job-name=waggle-cache-clips
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=runs/train/%x-%j.out
#SBATCH --error=runs/train/%x-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1

module load cuda

cd /nobackup/archive/usr/adameuba/waggle_chat
mkdir -p runs/train

ENV_PREFIX="${ENV_PREFIX:-/home/adameuba/.conda/envs/waggle_env}"

if [ -z "${SOURCE_RUN:-}" ]; then
  echo "Set SOURCE_RUN to a directory with train_manifest.csv and val_manifest.csv (e.g. runs/train/11143630)" >&2
  exit 1
fi

CACHE_ROOT="${CLIP_CACHE_ROOT:-/tmp/${SLURM_JOB_ID}/waggle_clip_cache}"
mkdir -p "$CACHE_ROOT"

echo "SOURCE_RUN=$SOURCE_RUN"
echo "CACHE_ROOT=$CACHE_ROOT (ephemeral on this node; copy manifests + cache elsewhere if you need them after the job)"
echo

mamba run -p "$ENV_PREFIX" python cache_clips.py \
  --manifest "$SOURCE_RUN/train_manifest.csv" \
  --out_manifest "$SOURCE_RUN/train_manifest_cached.csv" \
  --cache_dir "$CACHE_ROOT"

mamba run -p "$ENV_PREFIX" python cache_clips.py \
  --manifest "$SOURCE_RUN/val_manifest.csv" \
  --out_manifest "$SOURCE_RUN/val_manifest_cached.csv" \
  --cache_dir "$CACHE_ROOT"

echo "Done. Train with:"
echo "  torchrun ... train.py --no_build_manifest \\"
echo "    --train_manifest $SOURCE_RUN/train_manifest_cached.csv \\"
echo "    --val_manifest $SOURCE_RUN/val_manifest_cached.csv \\"
echo "    --out_dir runs/train/<new_job> ..."
