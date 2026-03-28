#!/bin/bash
# BYU RC: GPUs via Slurm only. Submit: sbatch slurm_train_full.sbatch  or  ./run_with_logs.sh
#SBATCH --job-name=waggle-train
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=l40s:1
#SBATCH --output=runs/full/%x-%j.out
#SBATCH --error=runs/full/%x-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1

module load cuda

cd /nobackup/archive/usr/adameuba/waggle_chat
mkdir -p runs/full

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-?}"
echo "node=$(hostname)"

mamba run -p /home/adameuba/.conda/envs/waggle_env python -u train.py \
  --data_dir data \
  --out_dir runs/full \
  --epochs 30 \
  --batch_size 4 \
  --lr 1e-4 \
  --weight_decay 1e-2 \
  --duration_loss_weight 0.5 \
  --window_seconds 3 \
  --fps 30 \
  --clip_frames 32 \
  --stride 2 \
  --seed 0

ls -lh runs/full/*.pt runs/full/*manifest.csv 2>/dev/null || true
