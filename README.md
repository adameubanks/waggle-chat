## Waggle detection + duration

Center-frame clips → **waggle vs not** (BCE) + **duration** on positives (log-space L1). Inference slides centers; CSV columns are frame indices.

### Setup

```bash
mamba env update -p /home/adameuba/.conda/envs/waggle_env -f environment.yml
```

Data: `data/annotations/filtered_waggles_*.csv`, `data/raw_videos/*.mp4`. `.gitignore` skips `runs/`, checkpoints, raw video.

### Slurm (BYU RC / GPU nodes)

Batch script: **`train.sbatch`** (multi-GPU training, 30 epochs). From repo root:

```bash
./run_with_logs.sh train.sbatch
# or: sbatch train.sbatch
```

**Where the logs are:** Slurm writes stdout/stderr under **`runs/train/`**, not the repo root. After you submit, note the job id (`squeue -u "$USER"` or the line printed by `sbatch --parsable`). Files look like:

- `runs/train/waggle-train-<jobid>.out`
- `runs/train/waggle-train-<jobid>.err`

(`waggle-train` comes from `#SBATCH --job-name` in `train.sbatch`.) That directory is **gitignored**, so you will not see those logs in `git status`; they are still on disk next to your clone.

`Ctrl+C` only stops `tail` in `run_with_logs.sh`; the job keeps running until you `scancel <jobid>`. Override env: `ENV_PREFIX=/path/to/env sbatch train.sbatch`.

**Quick local smoke (CPU, tiny):** see § Local below — useful before burning GPU hours.

### If you see huge `core` / `core.*` files in the project root

Those are **core dumps** (a C/C++/CUDA library or the Python interpreter crashed hard: segfault, bad CUDA state, OOM from the OS’s point of view, etc.). They are unrelated to normal Python tracebacks in the `.err` log.

- Inspect the **`.err`** file for the same job time window.
- Safe to delete: `rm -f core core.*`
- To **stop creating** cores while debugging: `ulimit -c 0` (in your shell, or add `ulimit -c 0` near the top of `train.sbatch` after the shebang).

### Local (CPU, slow)

```bash
CUDA_VISIBLE_DEVICES= python train.py --data_dir data --out_dir runs/local --epochs 2 --batch_size 2
```

### Infer & overlay

```bash
python infer.py --video data/raw_videos/WaggleDance_36.mp4 --ckpt runs/train/<jobid>/best.pt --out_csv predicted_36.csv --infer_stride_seconds 0.5
python overlay.py --video data/raw_videos/WaggleDance_36.mp4 --segments_csv predicted_36.csv --out_video overlay.mp4
```
