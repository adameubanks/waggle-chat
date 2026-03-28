#!/bin/bash
# Live logs: ./run_with_logs.sh   (default submits this file)
#SBATCH --job-name=waggle-train
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# BYU: many A100 pools are preempt-only; L40S/H200 are normal QoS (no standby).
#SBATCH --gpus=l40s:1
#SBATCH --output=runs/full/%x-%j.out
#SBATCH --error=runs/full/%x-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1

module load cuda

cd /nobackup/archive/usr/adameuba/waggle_chat
mkdir -p runs/full

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-?}"
echo "logs: ${SLURM_SUBMIT_DIR:-$PWD}/runs/full/waggle-train-${SLURM_JOB_ID:-?}.out"
echo "node=$(hostname)"

mamba run -p /home/adameuba/.conda/envs/waggle_env python -u train.py \
  --data_dir data \
  --out_dir runs/full \
  --epochs 30 \
  --batch_size 8 \
  --lr 3e-4 \
  --window_seconds 3 \
  --fps 30 \
  --clip_frames 32 \
  --stride 2 \
  --seed 0 \
  --threshold_sweep

ls -lh runs/full/*.pt runs/full/*manifest.csv 2>/dev/null || true
