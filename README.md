## Waggle detection + duration

Pipeline learns **waggle vs background** (video windows) and **waggle run duration (seconds)** from `filtered_waggles_*.csv` (`duration` column = distance proxy) + `data/raw_videos/*.mp4`. **Bearing** is deferred to classical CV vs annotations.

`.gitignore` excludes **`runs/`**, **checkpoints (`*.pt`)**, **`data/raw_videos/`**, and **video files** so the repo stays small on GitHub. Put MP4s locally; uncomment `data/annotations/` in `.gitignore` if CSVs must not be public.

### Setup with conda

```bash
conda create -n waggle python=3.11 -y
conda activate waggle

conda install -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.1 -y
pip install numpy pandas tqdm
```

### Train on BYU HPC (GPU — use Slurm)

GPUs are only available inside **Slurm batch jobs** on compute nodes. Running `python train.py` on a login node will not give you a usable GPU (and may error with CUDA busy/unavailable).

From the repo root on the cluster:

```bash
./run_with_logs.sh
# same as: sbatch slurm_train_full.sbatch
# then: tail -f runs/full/waggle-train-<JOBID>.out
```

Edit `slurm_train_full.sbatch` if you need a different GPU type, walltime, or `mamba run -p` path to your env.

**Training (`train.py`):** plain **BCE** (waggle vs not, `pos_weight` from class counts), **L1 in log-duration space** on waggle windows only, combined with `--duration_loss_weight`. **AdamW** + **cosine LR** to `lr/100`. Logs: `val_bce`, `val_dur_l1(log)`, `val_dur_mae_s` (seconds), `val_total`, P/R/F1, plus **`prob_mean`** / **`pos_rate`** (sanity vs all-0 or all-1 logits).

Outputs under `--out_dir` (Slurm default `runs/full/`): manifests, `best.pt` / `last.pt` (min `val_total`).

### Train locally (CPU-only, slow / debug)

```bash
CUDA_VISIBLE_DEVICES= python train.py --data_dir data --out_dir runs/local --epochs 2 --batch_size 2
```

### Infer

```bash
python infer.py --video data/raw_videos/WaggleDance_36.mp4 --ckpt runs/full/best.pt --out_csv predicted_36.csv
```

CSV: `startFrame,endFrame,score,duration_s`.

### Validate detection + duration vs annotations

```bash
python validate_segments.py --gt_csv data/annotations/filtered_waggles_WaggleDance_36.csv --pred_csv predicted_36.csv
```

Reports segment-level precision/recall/F1 (IoU match) and **duration MAE (s)** on matched segments.
