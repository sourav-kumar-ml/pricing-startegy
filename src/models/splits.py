from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.utils.paths import PROCESSED_DIR, ensure_dir


DATE_COLUMN = "CALENDAR_DATE"
TARGET_COLUMN = "log_Q"


@dataclass
class SplitManifestEntry:
    sell_id: int
    train_path: str
    val_path: str
    train_rows: int
    val_rows: int
    train_date_min: str
    train_date_max: str
    val_date_min: str
    val_date_max: str


def time_based_split(df: pd.DataFrame, ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between 0 and 1")
    df_sorted = df.sort_values(DATE_COLUMN)
    split_idx = int(len(df_sorted) * ratio)
    split_idx = max(1, min(split_idx, len(df_sorted) - 1))  # ensure non-empty splits
    train = df_sorted.iloc[:split_idx].copy()
    val = df_sorted.iloc[split_idx:].copy()
    return train, val


def load_features_with_ids(path: Path | None = None) -> pd.DataFrame:
    path = path or (PROCESSED_DIR / "features_with_ids.parquet")
    df = pd.read_parquet(path)
    if DATE_COLUMN not in df.columns or "SELL_ID" not in df.columns:
        raise ValueError(f"Expected {path} to include {DATE_COLUMN} and SELL_ID columns.")
    return df


def split_by_sku(
    df: pd.DataFrame, ratio: float = 0.8, output_dir: Path | None = None
) -> Dict[int, SplitManifestEntry]:
    output_dir = output_dir or (PROCESSED_DIR / "splits")
    ensure_dir(output_dir)
    manifest: Dict[int, SplitManifestEntry] = {}
    for sell_id, group in df.groupby("SELL_ID"):
        train, val = time_based_split(group, ratio=ratio)
        train_path = output_dir / f"{sell_id}_train.parquet"
        val_path = output_dir / f"{sell_id}_val.parquet"
        train.to_parquet(train_path, index=False)
        val.to_parquet(val_path, index=False)
        manifest[int(sell_id)] = SplitManifestEntry(
            sell_id=int(sell_id),
            train_path=str(train_path),
            val_path=str(val_path),
            train_rows=len(train),
            val_rows=len(val),
            train_date_min=str(train[DATE_COLUMN].min()),
            train_date_max=str(train[DATE_COLUMN].max()),
            val_date_min=str(val[DATE_COLUMN].min()),
            val_date_max=str(val[DATE_COLUMN].max()),
        )
    return manifest


def save_manifest(manifest: Dict[int, SplitManifestEntry], path: Path | None = None) -> Path:
    path = path or (PROCESSED_DIR / "splits" / "manifest.json")
    ensure_dir(path.parent)
    serializable: List[dict] = [asdict(entry) for entry in manifest.values()]
    path.write_text(json.dumps(serializable, indent=2) + "\n")
    return path


def run_splits(ratio: float = 0.8) -> Path:
    df = load_features_with_ids()
    manifest = split_by_sku(df, ratio=ratio)
    return save_manifest(manifest)


if __name__ == "__main__":
    path = run_splits()
    print(f"Wrote split manifest to {path}")
