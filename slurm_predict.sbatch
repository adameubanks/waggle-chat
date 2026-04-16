#!/bin/bash
#SBATCH --job-name=waggle-predict
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --output=runs/slurm/%x-%j.out
#SBATCH --error=runs/slurm/%x-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

module load cuda

cd /nobackup/archive/usr/adameuba/waggle_chat
mkdir -p runs/slurm runs/predict

ENV_PREFIX="${ENV_PREFIX:-/home/adameuba/.conda/envs/waggle_env}"
VIDEO="${VIDEO:?set VIDEO=/path/to/video.mp4}"
CKPT="${CKPT:?set CKPT=/path/to/best.pt}"
OUT_CSV="${OUT_CSV:-runs/predict/${SLURM_JOB_ID:-local}.csv}"

echo "job=${SLURM_JOB_ID:-?} node=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "VIDEO=$VIDEO"
echo "CKPT=$CKPT"
echo "OUT_CSV=$OUT_CSV"

mamba run -p "$ENV_PREFIX" python run.py predict \
  --video "$VIDEO" \
  --ckpt "$CKPT" \
  --out_csv "$OUT_CSV" \
  --fps 30 \
  --clip_frames 16 \
  --stride 2 \
  --infer_step_frames 8 \
  --thr_on 0.6 \
  --thr_off 0.4 \
  --min_len_frames 8 \
  --gap_frames 8 \
  --device "cuda:0"

ls -lah "$OUT_CSV" || true

