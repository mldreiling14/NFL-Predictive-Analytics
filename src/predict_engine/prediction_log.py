import sqlite3
from datetime import datetime, timezone
import pandas as pd


def _ensure_table(conn):
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


def log_predictions(predictions_df, db_path, season, week, source='auto'):
    """
    Freezes each game's prediction the first time it's seen, via INSERT OR
    IGNORE on game_id (the primary key). predict_week() gets called on
    every page load all week using the same snapshots, so later calls are
    silently no-ops here - only the first write for a given game sticks.
    """
    if predictions_df.empty:
        return
    conn = sqlite3.connect(db_path)
    _ensure_table(conn)
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            g['game_id'], season, week, g['home_team'], g['away_team'],
            float(g['home_win_prob']), float(g['away_win_prob']),
            source, now
        )
        for _, g in predictions_df.iterrows()
    ]
    conn.executemany(
        """INSERT OR IGNORE INTO logged_predictions
           (game_id, season, week, home_team, away_team, home_win_prob, away_win_prob, source, logged_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows
    )
    conn.commit()
    conn.close()


def get_logged_predictions(db_path, season=None, week=None):
    conn = sqlite3.connect(db_path)
    _ensure_table(conn)
    query = "SELECT * FROM logged_predictions WHERE 1=1"
    params = []
    if season is not None:
        query += " AND season = ?"
        params.append(season)
    if week is not None:
        query += " AND week = ?"
        params.append(week)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def get_logged_prediction_for_game(game_id, db_path):
    conn = sqlite3.connect(db_path)
    _ensure_table(conn)
    row = conn.execute(
        "SELECT home_win_prob, away_win_prob FROM logged_predictions WHERE game_id = ?",
        (game_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {'home_win_prob': row[0], 'away_win_prob': row[1]}