"""
Rebuilds the pre-built snapshot cache the web app reads at startup.

Run from the project root after each week's games finish (and after updating
the local database if needed):

    python scripts/build_snapshots_cache.py

then commit and push data/snapshots/ so Render picks it up.
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from predict_engine.snapshots import build_snapshots
from predict_engine.snapshot_cache import SNAPSHOT_SEASONS, save_snapshots

DB_PATH = os.path.join(ROOT, "data", "nfl.db")
CACHE_DIR = os.path.join(ROOT, "data", "snapshots")

if __name__ == "__main__":
    snapshots = build_snapshots(db_path=DB_PATH, seasons=SNAPSHOT_SEASONS)
    save_snapshots(snapshots, CACHE_DIR)
    total_kb = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in os.listdir(CACHE_DIR)) / 1024
    print(f"Saved {len(snapshots)} tables to {os.path.normpath(CACHE_DIR)} ({total_kb:.0f} KB)")