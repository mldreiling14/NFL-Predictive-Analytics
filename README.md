# NFL Win Probability Model

A machine learning pipeline that predicts NFL game outcomes as a win probability, trained on real historical data (2015 through the current season — see `src/config.py`) and capable of generating live predictions for upcoming games. Includes a web app for browsing each week's predictions, with a Vegas odds comparison, a per-game detail page, and in-game live win probability.

**Model accuracy:** ~67% in a week-by-week walk-forward backtest over 2024–2026 (663 games, each week predicted by a model trained only on earlier games), versus ~67% for the Vegas favorite on the same games. See [`FEATURES.md`](./FEATURES.md) for the full technical breakdown of every feature, what was tried and rejected, and known limitations. See [`HOW_IT_WORKS.md`](./HOW_IT_WORKS.md) for a plain-English, non-technical explanation.

---

## What's in this repo

```
sports_betting/
├── data/
│   ├── nfl.db              # SQLite database (created by fetch_data.py)
│   └── snapshots/          # Pre-built "current state" cache the web app loads at startup
├── models/
│   ├── win_probability_model.joblib       # Pre-game model (created by train_model.py)
│   ├── live_wp_model.joblib               # In-game live win probability model
│   └── prob_to_spread_calibration.joblib  # Converts a win probability to a point spread
├── retrain_reports/    # JSON reports from the weekly retrain pipeline
├── notebooks/          # Exploration and testing notebooks
├── scripts/
│   └── build_snapshots_cache.py  # Rebuilds data/snapshots/
├── src/
│   ├── config.py           # Which seasons to pull/train on (update once a year)
│   ├── fetch_data.py       # Pulls raw NFL schedules into SQLite
│   ├── features/           # Feature-engineering functions, one module per feature group
│   ├── build_features.py   # Runs the full feature pipeline, saves games_with_features
│   ├── train_model.py      # Trains and saves the production model
│   ├── retrain_pipeline.py # Weekly evaluate-then-promote retraining
│   ├── train_live_wp_model.py  # Trains the in-game live model
│   ├── predict_engine/     # Snapshots, matchup features, injuries, live data
│   └── predict.py          # Generates live predictions for upcoming games
├── web/                # FastAPI web app (the deployed site)
├── app.py              # Streamlit app (older local alternative)
├── requirements.txt
├── FEATURES.md          # Full technical feature documentation
└── HOW_IT_WORKS.md       # Plain-English project summary
```

---

## Setup

**1. Clone the repo and create a virtual environment**

```bash
git clone <your-repo-url>
cd sports_betting
python -m venv venv
```

Activate it:
- Windows (Git Bash): `source venv/Scripts/activate`
- Mac/Linux: `source venv/bin/activate`

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the pipeline

Run these in order. Each step builds on the one before it.

**1. Pull raw schedule data**

```bash
python src/fetch_data.py
```

Creates `data/nfl.db` with a `games` table covering every season in `src/config.py`, including the current season's full schedule. Games not yet played are kept, with no scores and no `home_win` label. Takes a minute or two.

**2. Build all engineered features**

```bash
python src/build_features.py
```

Pulls player stats, team stats, injuries, coaching data, and more from [`nflreadpy`](https://github.com/nflverse/nflreadpy), then runs every feature function in `src/features/`. Saves a `games_with_features` table back into `data/nfl.db`. This is the slowest step — expect several minutes, since it pulls multiple seasons of detailed player-level data.

**3. Train the model**

```bash
python src/train_model.py
```

Trains a logistic regression model on all completed games (unplayed games are excluded) and saves it to `models/win_probability_model.joblib`, bundled together with the imputer used to fill missing values and the exact list of features the model expects.

**4. Build the snapshot cache**

```bash
python scripts/build_snapshots_cache.py
```

Builds "current state" snapshots for every team (recent form, offense/defense ratings, current starters from the live depth charts, etc.) from completed games, and saves them as parquet files in `data/snapshots/`. The web app reads these at startup instead of rebuilding them on every boot. If the folder is missing, the app falls back to building them live, which is slow.

**5. Run the app**

```bash
uvicorn web.main:app --reload
```

Shows win probabilities for each upcoming week, with team colors, a Vegas odds comparison, and a "favorite" call-out for each game. Clicking a game opens a detail page with side-by-side team cards (starting QB/RB/WR, the QB's recent game log) and a same-day injury report from ESPN. During games, it also shows a live in-game win probability.

The older Streamlit version still works with `streamlit run app.py`.

---

## Updating for a new season or week

After each week's games are final:

```bash
python src/retrain_pipeline.py            # refresh data, train a candidate, compare it to production
python src/retrain_pipeline.py --promote  # only if the report looks good
python scripts/build_snapshots_cache.py   # refresh current form, starters, ratings
```

The first command re-pulls data, rebuilds features, trains a candidate model on everything before the most recently completed week, and scores both the candidate and the current production model on that week. The report goes to `retrain_reports/YYYY-MM-DD.json`. One week is only ~14–16 games, so a single report is noisy; look at several before promoting. `--promote` retrains on all completed games and overwrites `models/win_probability_model.joblib`. It does not commit or push.

Then commit `models/`, `data/nfl.db` and `data/snapshots/` and push. The deployed site only picks up new snapshots and models on deploy.

Once a year, bump `CURRENT_SEASON` in `src/config.py`.

---

## A few honest notes

- **This is not a betting system.** The model has not been shown to beat Vegas — when they disagree, Vegas is right more often than not. Treat predictions as a well-reasoned estimate, not a guarantee.
- **Live predictions use each team's most recent identified starters** (via official depth charts, not just "whoever played last") as of the last snapshot build. Rebuild the snapshot cache late in the week to catch QB changes. The model's injury flags come from nflverse's weekly injury reports (Out/Doubtful/Questionable), which lag same-day news; the ESPN injury report on the detail page is display-only.
- **Weather is tracked but not used in the model** — it showed a real, sensible effect in specific cases (e.g., warm-climate teams struggling in extreme cold) but wasn't statistically powerful enough to improve overall accuracy.

See `FEATURES.md` for the complete list of what's included, what was tested and rejected, and why.
