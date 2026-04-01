## Waggle detection + duration

Center-frame clips → **waggle vs not** (BCE) + **duration** on positives (log-space L1). Inference slides centers; CSV columns are frame indices.

### Setup

```bash
mamba env update -p /home/adameuba/.conda/envs/waggle_env -f environment.yml
```

Data: `data/annotations/filtered_waggles_*.csv`, `data/raw_videos/*.mp4`. `.gitignore` skips `runs/`, checkpoints, raw video.

### Slurm (BYU RC / GPU nodes)

Use batch jobs for GPU work. From repo root:

```bash
./run_with_logs.sh slurm_smoke.sbatch    # 1 GPU, 1 epoch + infer + overlay
./run_with_logs.sh slurm_train.sbatch    # multi-GPU training
```

`Ctrl+C` only stops `tail`; cancel with `scancel <jobid>`. Override env: `ENV_PREFIX=/path/to/env sbatch ...`.

**Decode once (repeat training):** After manifests exist under `runs/train/<jobid>/`, run `slurm_cache_clips.sbatch` with `SOURCE_RUN` set. Cached CSVs use column `clip_pt`. Train with `--no_build_manifest --train_manifest ..._cached.csv --val_manifest ..._cached.csv`. Fast scratch: node **`/tmp`** (not shared, not permanent); durable or multi-node: `CLIP_CACHE_ROOT` on shared storage.

### Local (CPU, slow)

```bash
CUDA_VISIBLE_DEVICES= python train.py --data_dir data --out_dir runs/local --epochs 2 --batch_size 2
```

### Infer & overlay

```bash
python infer.py --video data/raw_videos/WaggleDance_36.mp4 --ckpt runs/train/<jobid>/best.pt --out_csv predicted_36.csv --infer_stride_seconds 0.5
python overlay.py --video data/raw_videos/WaggleDance_36.mp4 --segments_csv predicted_36.csv --out_video overlay.mp4
```
