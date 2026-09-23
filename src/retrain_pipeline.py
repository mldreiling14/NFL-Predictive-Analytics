"""
Weekly retrain-and-evaluate pipeline for the pre-game win probability model.

Two modes:

  python src/retrain_pipeline.py
      Default "evaluate" mode. Meant to run unattended (e.g. weekly via
      Windows Task Scheduler). Re-pulls data, rebuilds features, trains a
      CANDIDATE model, and compares it against the CURRENT production model
      on a genuinely-unseen held-out slice (the most recently completed
      week of games - data the current production model was trained
      before, so this is a fair, leakage-free comparison for both models).
      Writes a timestamped report and saves the candidate model to a
      staging path. Does NOT touch the live model or git - nothing here
      can affect the deployed site.

  python src/retrain_pipeline.py --promote
      Run this yourself, after reading a report and deciding the numbers
      look good. Retrains on ALL available data (matching how the
      production model has always been trained - see train_model.py) and
      overwrites the real production model file. Then prints the exact git
      commands to run - it does not run them for you, since `git push` is
      what actually triggers your Render deploy, and that should stay a
      conscious decision, not something a scheduled task does by itself.

Why the held-out slice is "the most recent completed week" rather than a
random 80/20 split: the current production model was already trained on
nearly all of your historical data, so a random split would hand it a
"test" set it has effectively already seen, making it look artificially
good regardless of whether retraining actually helps. The most recent
week's games didn't exist yet when the current model was trained, so
they're honestly unseen for BOTH the current model and the candidate -
this is a real answer to "would retraining on what just happened help."
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import joblib
import pandas as pd
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_model import FEATURE_COLS, train_and_save_model  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DB_PATH = os.path.join(REPO_ROOT, "data", "nfl.db")
MODEL_PATH = os.path.join(REPO_ROOT, "models", "win_probability_model.joblib")
CANDIDATE_MODEL_PATH = os.path.join(REPO_ROOT, "models", "candidate_model.joblib")
REPORTS_DIR = os.path.join(REPO_ROOT, "retrain_reports")


def refresh_data():
    """Re-pulls raw data and rebuilds games_with_features, the same two
    steps you'd run by hand - just automated here so the scheduled run
    always evaluates against the freshest completed games."""
    print("Refreshing data: fetch_data.py ...")
    subprocess.run([sys.executable, os.path.join(REPO_ROOT, "src", "fetch_data.py")], check=True)
    print("Refreshing data: build_features.py ...")
    subprocess.run([sys.executable, os.path.join(REPO_ROOT, "src", "build_features.py")], check=True)


def load_games():
    conn = sqlite3.connect(DB_PATH)
    df_full = pd.read_sql_query("SELECT * FROM games_with_features", conn)
    conn.close()
    df_full["home_win"] = (df_full["home_score"] > df_full["away_score"]).astype(int)
    return df_full


def split_most_recent_week(df_full):
    """Test set = the single most recently completed (season, week).
    Train set = everything strictly before it. Ties season/week together
    so this is correct across a season boundary too."""
    completed = df_full.dropna(subset=["home_score"])
    latest_season, latest_week = completed[["season", "week"]].sort_values(
        ["season", "week"]
    ).iloc[-1]

    test_mask = (df_full["season"] == latest_season) & (df_full["week"] == latest_week)
    test_df = df_full[test_mask]
    train_df = df_full[~test_mask & df_full["home_score"].notna()]
    return train_df, test_df, int(latest_season), int(latest_week)


def fit_candidate(train_df):
    X = train_df[FEATURE_COLS].copy()
    y = train_df["home_win"]

    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=FEATURE_COLS, index=X.index)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_imputed, y)

    return {"model": model, "imputer": imputer, "feature_cols": FEATURE_COLS}


def evaluate_bundle(bundle, test_df):
    X = test_df[bundle["feature_cols"]].copy()
    y = test_df["home_win"]

    X_imputed = pd.DataFrame(
        bundle["imputer"].transform(X), columns=bundle["feature_cols"], index=X.index
    )
    probs = bundle["model"].predict_proba(X_imputed)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "n_games": len(test_df),
        "accuracy": accuracy_score(y, preds),
        "log_loss": log_loss(y, probs, labels=[0, 1]),
        "brier": brier_score_loss(y, probs),
    }


def format_metrics(label, m):
    return (
        f"{label}: n={m['n_games']}  "
        f"accuracy={m['accuracy']:.4f}  "
        f"log_loss={m['log_loss']:.4f}  "
        f"brier={m['brier']:.4f}"
    )


def run_evaluate():
    refresh_data()
    df_full = load_games()
    train_df, test_df, season, week = split_most_recent_week(df_full)

    if len(test_df) < 4:
        print(
            f"Only {len(test_df)} completed games in the most recent week "
            f"({season} week {week}) - too small a sample to draw a real "
            f"conclusion from. Skipping evaluation this run; the report "
            f"below is informational only."
        )

    print(f"Held-out test week: {season} week {week} ({len(test_df)} games)")
    print(f"Training candidate on {len(train_df)} games ...")
    candidate_bundle = fit_candidate(train_df)

    if not os.path.exists(MODEL_PATH):
        print(f"No existing production model found at {MODEL_PATH} - nothing to compare against.")
        current_metrics = None
    else:
        current_bundle = joblib.load(MODEL_PATH)
        current_metrics = evaluate_bundle(current_bundle, test_df)

    candidate_metrics = evaluate_bundle(candidate_bundle, test_df)

    print()
    if current_metrics:
        print(format_metrics("Current production model", current_metrics))
    print(format_metrics("Candidate (retrained) model", candidate_metrics))

    os.makedirs(os.path.dirname(CANDIDATE_MODEL_PATH), exist_ok=True)
    joblib.dump(candidate_bundle, CANDIDATE_MODEL_PATH)
    print(f"\nCandidate model saved to {CANDIDATE_MODEL_PATH} (staging only - not live).")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "run_at": datetime.now().isoformat(),
                "held_out_season": season,
                "held_out_week": week,
                "n_train_games": len(train_df),
                "current_production_model": current_metrics,
                "candidate_model": candidate_metrics,
            },
            f,
            indent=2,
        )
    print(f"Report written to {report_path}")
    print(
        "\nNothing has been deployed. Review the numbers above, and if the "
        "candidate looks good, run:\n"
        "    python src/retrain_pipeline.py --promote"
    )


def run_promote():
    print("Retraining production model on ALL available data ...")
    train_and_save_model(db_path=DB_PATH, model_path=MODEL_PATH)

    if os.path.exists(CANDIDATE_MODEL_PATH):
        os.remove(CANDIDATE_MODEL_PATH)

    print(
        "\nProduction model updated locally. To deploy it, review the diff "
        "and run:\n\n"
        f"    git add {os.path.relpath(MODEL_PATH, REPO_ROOT)} "
        f"{os.path.relpath(os.path.join(REPO_ROOT, 'data', 'nfl.db'), REPO_ROOT)}\n"
        '    git commit -m "Weekly model retrain"\n'
        "    git push\n\n"
        "git push is what triggers your Render deploy - nothing here does "
        "that automatically."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true", help="Retrain on all data and update the production model file.")
    args = parser.parse_args()

    if args.promote:
        run_promote()
    else:
        run_evaluate()