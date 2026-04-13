#!/bin/bash
#SBATCH --job-name=waggle-train
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --gpus=4
#SBATCH --output=runs/train/%x-%j.out
#SBATCH --error=runs/train/%x-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

module load cuda

cd /nobackup/archive/usr/adameuba/waggle_chat
mkdir -p runs/train

ENV_PREFIX="${ENV_PREFIX:-/home/adameuba/.conda/envs/waggle_env}"
MANIFEST_DIR="${MANIFEST_DIR:-runs/train/11165731}"
ROOT_OUTDIR="runs/train/${SLURM_JOB_ID:-local}"
mkdir -p "$ROOT_OUTDIR"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -cve '^$')
else
  NPROC="${SLURM_GPUS_ON_NODE:-1}"
fi
NPROC="${NPROC:-1}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-?}"
echo "node=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "NPROC=$NPROC"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "MANIFEST_DIR=$MANIFEST_DIR"
echo "ROOT_OUTDIR=$ROOT_OUTDIR"
echo

for DUR_W in 0.2 0.4 0.6; do
  OUTDIR="$ROOT_OUTDIR/dur_w_${DUR_W}"
  mkdir -p "$OUTDIR"
  echo "OUTDIR=$OUTDIR"
  mamba run -p "$ENV_PREFIX" torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" train.py \
    --data_dir data \
    --out_dir "$OUTDIR" \
    --no_build_manifest \
    --train_manifest "$MANIFEST_DIR/train_manifest.csv" \
    --val_manifest "$MANIFEST_DIR/val_manifest.csv" \
    --epochs 30 \
    --batch_size 32 \
    --lr 3e-4 \
    --weight_decay 5e-3 \
    --duration_loss_weight "$DUR_W" \
    --center_stride_frames 1 \
    --neg_per_pos 3 \
    --fps 30 \
    --clip_frames 32 \
    --stride 2 \
    --seed 0 \
    --num_workers 3 \
    --prefetch_factor 2 \
    --val_num_workers 2 \
    --val_every 2 \
    --select_metric best_f1
done

ls -lh "$ROOT_OUTDIR"/**/*.pt 2>/dev/null || true
ls -lh "$MANIFEST_DIR"/*manifest.csv 2>/dev/null || true
