## WaggleChat (waggle detector)

Minimal pipeline:

- **Train** a binary waggle detector on annotated windows.
- **Predict** waggle segments + duration via thresholding + hysteresis.
- **Direction** (optional): estimate a dominant motion axis with optical flow (`waggle/direction.py`).

Data expected:

- `data/annotations/filtered_waggles_*.csv`
- `data/raw_videos/*.mp4`

### Setup

```bash
mamba env update -p /home/adameuba/.conda/envs/waggle_env -f environment.yml
```

If `cv2` fails to import (libjpeg/libtiff symbol errors), recreate the env so all libs come from `conda-forge`:

```bash
mamba env remove -p /home/adameuba/.conda/envs/waggle_env
mamba env create -p /home/adameuba/.conda/envs/waggle_env -f environment.yml
```

### Slurm

- **Train**:

```bash
sbatch slurm_train.sbatch
```

Outputs under `runs/train/<jobid>/`:

- `best.pt`, `last.pt`
- `manifests/train_manifest.csv`, `manifests/val_manifest.csv`

- **Predict**:

```bash
VIDEO=data/raw_videos/WaggleDance_36.mp4 CKPT=runs/train/<jobid>/best.pt sbatch slurm_predict.sbatch
```

### Local (quick smoke)

```bash
python run.py prep --data_dir data --out_dir runs/manifests
python run.py train --train_manifest runs/manifests/train_manifest.csv --val_manifest runs/manifests/val_manifest.csv --out_dir runs/train/local --epochs 2
python run.py predict --video data/raw_videos/WaggleDance_36.mp4 --ckpt runs/train/local/best.pt --out_csv runs/predict/local.csv
python overlay.py --video data/raw_videos/WaggleDance_36.mp4 --segments_csv runs/predict/local.csv --out_video runs/predict/overlay.mp4
```
