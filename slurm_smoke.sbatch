#!/bin/bash
#SBATCH --job-name=waggle-smoke
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --output=runs/smoke/%x-%j.out
#SBATCH --error=runs/smoke/%x-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

module load cuda

cd /nobackup/archive/usr/adameuba/waggle_chat
mkdir -p runs/smoke

ENV_PREFIX="${ENV_PREFIX:-/home/adameuba/.conda/envs/waggle_env}"
OUTDIR="runs/smoke/${SLURM_JOB_ID:-local}"
mkdir -p "$OUTDIR"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-?}"
echo "node=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "OUTDIR=$OUTDIR"
echo

echo "=== GPU ==="
nvidia-smi -L || true
echo

echo "=== torch sanity ==="
mamba run -p "$ENV_PREFIX" python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
echo

echo "=== train (1 epoch) ==="
mamba run -p "$ENV_PREFIX" python train.py \
  --data_dir data \
  --out_dir "$OUTDIR" \
  --epochs 1 \
  --batch_size 4 \
  --max_samples 2048 \
  --center_stride_frames 4 \
  --lr 1e-4 \
  --weight_decay 1e-2 \
  --duration_loss_weight 0.5 \
  --neg_per_pos 3 \
  --fps 30 \
  --clip_frames 32 \
  --stride 2 \
  --seed 0 \
  --num_workers 2
echo

echo "=== manifest stats ==="
OUTDIR="$OUTDIR" mamba run -p "$ENV_PREFIX" python - <<'PY'
import os
from pathlib import Path
import pandas as pd

outdir = Path(os.environ["OUTDIR"])
train = pd.read_csv(outdir / "train_manifest.csv")
val = pd.read_csv(outdir / "val_manifest.csv")
def s(df):
    n = int(len(df))
    pos = int(df["is_waggle"].sum())
    return n, pos, (pos / n if n else 0.0)
tn,tp,tr = s(train)
vn,vp,vr = s(val)
print(f"train: n={tn} pos={tp} pos_rate={tr:.4f}")
print(f"val:   n={vn} pos={vp} pos_rate={vr:.4f}")
print("ckpts:", (outdir / "best.pt").exists(), (outdir / "last.pt").exists())
print("example video:", val.iloc[0]['video'])
PY
echo

VIDEO="$(OUTDIR="$OUTDIR" mamba run -p "$ENV_PREFIX" python - <<'PY'
import os
from pathlib import Path
import pandas as pd
outdir = Path(os.environ["OUTDIR"])
val = pd.read_csv(outdir / "val_manifest.csv")
print(val.iloc[0]["video"])
PY
)"

echo "=== infer ==="
mamba run -p "$ENV_PREFIX" python infer.py \
  --video "$VIDEO" \
  --ckpt "$OUTDIR/best.pt" \
  --out_csv "$OUTDIR/predicted_waggles.csv" \
  --fps 30 \
  --infer_stride_seconds 0.5 \
  --clip_frames 32 \
  --stride 2
echo

echo "=== overlay ==="
mamba run -p "$ENV_PREFIX" python overlay.py \
  --video "$VIDEO" \
  --segments_csv "$OUTDIR/predicted_waggles.csv" \
  --out_video "$OUTDIR/overlay.mp4"
echo

echo "=== outputs ==="
ls -lh "$OUTDIR"/* 2>/dev/null || true
