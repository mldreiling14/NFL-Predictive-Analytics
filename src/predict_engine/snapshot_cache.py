"""
Pre-built snapshot cache.

build_snapshots() downloads ~10 nflverse tables and runs rolling averages over
them. That is fine on a laptop but slow, and memory-hungry enough to hit
Render's free-tier limit, and it ran on every app boot (including every wake-up
from idle). Instead, run scripts/build_snapshots_cache.py locally after each
week's games, commit data/snapshots/, and the web app just reads small parquet
files at startup.

Parquet (not pickle) on purpose: pickles are tied to the exact pandas version
that wrote them, and Render may install a different one than your laptop.
"""
import json
import os
import time

import pandas as pd

from .snapshots import build_snapshots

SNAPSHOT_SEASONS = list(range(2024, 2027))
META_FILE = "meta.json"


def save_snapshots(snapshots, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    for name, df in snapshots.items():
        df.reset_index(drop=True).to_parquet(os.path.join(cache_dir, f"{name}.parquet"), index=False)
    meta = {"built_at": time.time(), "seasons": SNAPSHOT_SEASONS, "tables": sorted(snapshots)}
    with open(os.path.join(cache_dir, META_FILE), "w") as f:
        json.dump(meta, f, indent=2)


def load_snapshots(cache_dir):
    """Returns the snapshot dict, or None if the cache is missing/incomplete."""
    meta_path = os.path.join(cache_dir, META_FILE)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    snapshots = {}
    for name in meta["tables"]:
        path = os.path.join(cache_dir, f"{name}.parquet")
        if not os.path.exists(path):
            return None
        snapshots[name] = pd.read_parquet(path)
    age_hours = (time.time() - meta["built_at"]) / 3600
    print(f"Loaded {len(snapshots)} snapshot tables from cache (built {age_hours:.1f} hours ago)")
    return snapshots


def load_or_build_snapshots(db_path, cache_dir):
    """Fast path: read the pre-built cache. Fallback: build live (the old, slow behavior)."""
    snapshots = load_snapshots(cache_dir)
    if snapshots is not None:
        return snapshots
    print("No snapshot cache found - building live (slow). Run scripts/build_snapshots_cache.py to speed this up.")
    return build_snapshots(db_path=db_path, seasons=SNAPSHOT_SEASONS)