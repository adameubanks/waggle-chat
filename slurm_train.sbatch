#!/bin/bash
#SBATCH --job-name=waggle-train
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
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
OUTDIR="runs/train/${SLURM_JOB_ID:-local}"
mkdir -p "$OUTDIR"

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
echo "OUTDIR=$OUTDIR"
echo

mamba run -p "$ENV_PREFIX" torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" train.py \
  --data_dir data \
  --out_dir "$OUTDIR" \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-4 \
  --weight_decay 1e-2 \
  --duration_loss_weight 0.5 \
  --center_stride_frames 1 \
  --neg_per_pos 3 \
  --fps 30 \
  --clip_frames 32 \
  --stride 2 \
  --seed 0 \
  --num_workers 4

ls -lh "$OUTDIR"/*.pt "$OUTDIR"/*manifest.csv 2>/dev/null || true
