#!/bin/bash
#SBATCH --job-name=waggle-export-demo
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --output=runs/export/%x-%j.out
#SBATCH --error=runs/export/%x-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

module load cuda

cd /nobackup/archive/usr/adameuba/waggle_chat
mkdir -p runs/export

ENV_PREFIX="${ENV_PREFIX:-/home/adameuba/.conda/envs/waggle_env}"
VIDEO="${VIDEO:-data/raw_videos/WaggleDance_38.mp4}"
CKPT="${CKPT:-runs/train/11363514/dur_w_0.2/best.pt}"

OUT_DIR="${OUT_DIR:-docs}"
CLIP_ID="${CLIP_ID:-waggle38}"
TITLE="${TITLE:-WaggleDance_38 (best.pt)}"

FPS="${FPS:-30}"
INFER_STRIDE_SECONDS="${INFER_STRIDE_SECONDS:-0.5}"
CLIP_FRAMES="${CLIP_FRAMES:-32}"
STRIDE="${STRIDE:-2}"
THR_ON="${THR_ON:-0.6}"
THR_OFF="${THR_OFF:-0.4}"
METERS_PER_SECOND="${METERS_PER_SECOND:-300}"
ANGLE_OFFSET_DEG="${ANGLE_OFFSET_DEG:-0}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-?}"
echo "node=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "VIDEO=$VIDEO"
echo "CKPT=$CKPT"
echo "OUT_DIR=$OUT_DIR"
echo "CLIP_ID=$CLIP_ID"
echo

mamba run -p "$ENV_PREFIX" python export_demo.py \
  --video "$VIDEO" \
  --ckpt "$CKPT" \
  --out_dir "$OUT_DIR" \
  --clip_id "$CLIP_ID" \
  --title "$TITLE" \
  --fps "$FPS" \
  --infer_stride_seconds "$INFER_STRIDE_SECONDS" \
  --clip_frames "$CLIP_FRAMES" \
  --stride "$STRIDE" \
  --thr_on "$THR_ON" \
  --thr_off "$THR_OFF" \
  --meters_per_second "$METERS_PER_SECOND" \
  --angle_offset_deg "$ANGLE_OFFSET_DEG" \
  --device "cuda:0" \
  --copy_video

ls -lah "$OUT_DIR/data/manifest.json" "$OUT_DIR/data/runs/${CLIP_ID}.json" || true

