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

### Train

```bash
python train.py --data_dir data --out_dir runs/r3d18 --epochs 5
```

**Losses:** focal BCE (detection), SmoothL1 on `log1p(duration_s)` (waggle windows only). **Warmup:** `--duration_warmup_epochs` (cls-only first), `--lr_warmup_epochs`, `--grad_clip`. Logs: `val_cls`, `val_dur_sl1`, `val_dur_mae_s`, prec/rec/f1.

Outputs: `train_manifest.csv`, `val_manifest.csv`, `best.pt`, `last.pt`.

### Slurm + live logs (BYU)

```bash
./run_with_logs.sh
```

### Infer

```bash
python infer.py --video data/raw_videos/WaggleDance_36.mp4 --ckpt runs/r3d18/best.pt --out_csv predicted_36.csv
```

CSV: `startFrame,endFrame,score,duration_s`.

### Validate detection + duration vs annotations

```bash
python validate_segments.py --gt_csv data/annotations/filtered_waggles_WaggleDance_36.csv --pred_csv predicted_36.csv
```

Reports segment-level precision/recall/F1 (IoU match) and **duration MAE (s)** on matched segments.
