from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_splits(
    manifest: pd.DataFrame,
    *,
    out_dir: Path,
    train_frac: float = 0.8,
    seed: int = 0,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    vids = np.array(sorted(manifest["video"].unique()))
    rng.shuffle(vids)
    n_train = int(round(len(vids) * train_frac))
    train_vids = set(vids[:n_train].tolist())

    train_df = manifest[manifest["video"].isin(train_vids)].reset_index(drop=True)
    val_df = manifest[~manifest["video"].isin(train_vids)].reset_index(drop=True)

    train_path = out_dir / "train_manifest.csv"
    val_path = out_dir / "val_manifest.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print(
        f"[manifest] split -> {train_path.name}: {len(train_df)} rows, {train_df['video'].nunique()} videos | "
        f"{val_path.name}: {len(val_df)} rows, {val_df['video'].nunique()} videos",
        flush=True,
    )
    return train_path, val_path
