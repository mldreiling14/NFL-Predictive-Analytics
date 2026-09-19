import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import sqlite3
from datetime import datetime, timezone
import nflreadpy as nfl

# From the Render screenshot - team: win probability (%), not home/away ordered
screenshot_probs = {
    'NE': 25, 'SEA': 75,
    'SF': 36, 'LA': 64,
    'CHI': 60, 'CAR': 40,
    'TB': 34, 'CIN': 66,
    'NO': 28, 'DET': 72,
    'BUF': 53, 'HOU': 47,
    'BAL': 65, 'IND': 35,
    'CLE': 15, 'JAX': 85,
    'ATL': 45, 'PIT': 55,
    'NYJ': 44, 'TEN': 56,
    'ARI': 42, 'LAC': 58,
    'MIA': 72, 'LV': 28,
    'GB': 54, 'MIN': 46,
    'WAS': 38, 'PHI': 62,
    'DAL': 51, 'NYG': 49,
    'DEN': 55, 'KC': 45,
}

sched = nfl.load_schedules(seasons=[2026]).to_pandas()
week1 = sched[sched['week'] == 1]

db_path = os.path.join(os.path.dirname(__file__), "..", "data", "nfl.db")
conn = sqlite3.connect(db_path)
conn.execute("""
    CREATE TABLE IF NOT EXISTS logged_predictions (
        game_id TEXT PRIMARY KEY,
        season INTEGER NOT NULL,
        week INTEGER NOT NULL,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        home_win_prob REAL NOT NULL,
        away_win_prob REAL NOT NULL,
        source TEXT NOT NULL DEFAULT 'auto',
        logged_at TEXT NOT NULL
    )
""")

now = datetime.now(timezone.utc).isoformat()
inserted, missing = 0, []

for _, g in week1.iterrows():
    home, away = g['home_team'], g['away_team']
    if home not in screenshot_probs or away not in screenshot_probs:
        missing.append(g['game_id'])
        continue
    home_pct, away_pct = screenshot_probs[home], screenshot_probs[away]
    # normalize in case the two don't sum to exactly 100
    total = home_pct + away_pct
    home_prob, away_prob = home_pct / total, away_pct / total
    conn.execute(
        """INSERT OR IGNORE INTO logged_predictions
           (game_id, season, week, home_team, away_team, home_win_prob, away_win_prob, source, logged_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (g['game_id'], 2026, 1, home, away, home_prob, away_prob, 'manual_backfill', now)
    )
    inserted += 1

conn.commit()
conn.close()

print(f"Backfilled {inserted} Week 1 games.")
if missing:
    print(f"Could not match {len(missing)} games (team code mismatch?): {missing}")