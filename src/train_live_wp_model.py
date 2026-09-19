"""
Trains the in-game (live) win-probability model, separate from the
pre-game model in train_model.py. This model answers a different
question: given the CURRENT score, time remaining, and which team has
the ball, what's the home team's win probability right now?

Approach (see modeling-decisions.md for the fuller rationale): rather
than retroactively backtesting the pre-game model across history (which
would require rebuilding leakage-safe snapshots for every past game),
we train on the actual historical Vegas closing spread (spread_line),
which nflverse's play-by-play data already carries on every row. At
serving time, we convert our OWN pre-game model's win probability into
an equivalent spread using the calibration fit at the bottom of this
file, and feed that in as if it were the spread. This anchors the live
curve to our own model's kickoff number without needing to replay
history through the full feature pipeline.
"""
import numpy as np
import pandas as pd
import nflreadpy as nfl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import joblib

PBP_COLS = [
    'game_id', 'season', 'week', 'home_team', 'away_team', 'posteam',
    'game_seconds_remaining', 'total_home_score', 'total_away_score',
    'spread_line', 'result',
]

TOTAL_GAME_SECONDS = 3600.0


def build_training_data(seasons=range(2015, 2026)):
    frames = []
    for s in seasons:
        try:
            df = nfl.load_pbp(seasons=[s]).to_pandas()
        except Exception as e:
            print(f"Skipping {s}: {e}")
            continue
        frames.append(df[PBP_COLS].copy())
    pbp = pd.concat(frames, ignore_index=True)

    # Drop plays where possession or the clock isn't well-defined
    # (kickoffs, timeouts, end-of-quarter markers, etc.)
    pbp = pbp.dropna(subset=['posteam', 'game_seconds_remaining'])

    pbp['score_diff_home'] = pbp['total_home_score'] - pbp['total_away_score']
    pbp['time_remaining_frac'] = pbp['game_seconds_remaining'] / TOTAL_GAME_SECONDS
    pbp['possession_home'] = (pbp['posteam'] == pbp['home_team']).astype(int)

    # Interaction terms: let the model learn the handoff between
    # "pre-game strength matters most" (early) and "score/time matters
    # most" (late) rather than us hand-tuning fixed weights.
    pbp['spread_x_time_remaining'] = pbp['spread_line'] * pbp['time_remaining_frac']
    pbp['score_diff_x_time_elapsed'] = pbp['score_diff_home'] * (1 - pbp['time_remaining_frac'])

    pbp['home_win'] = (pbp['result'] > 0).astype(int)
    # Ties (result == 0) are genuinely ambiguous outcomes - drop rather
    # than force a 0/1 label onto them.
    pbp = pbp[pbp['result'] != 0]

    return pbp


FEATURE_COLS = [
    'score_diff_home', 'time_remaining_frac', 'possession_home',
    'spread_line', 'spread_x_time_remaining', 'score_diff_x_time_elapsed',
]


def train_live_wp_model(seasons=range(2015, 2026), model_path='models/live_wp_model.joblib'):
    pbp = build_training_data(seasons)

    # Split by game_id, not by row, so plays from the same game don't
    # leak between train and test (they're highly correlated).
    game_ids = np.asarray(pbp['game_id'].unique(), dtype=object)
    train_ids, test_ids = train_test_split(game_ids, test_size=0.15, random_state=42)

    train = pbp[pbp['game_id'].isin(train_ids)]
    test = pbp[pbp['game_id'].isin(test_ids)]

    X_train, y_train = train[FEATURE_COLS], train['home_win']
    X_test, y_test = test[FEATURE_COLS], test['home_win']

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    print(f"Trained on {len(train)} plays from {len(train_ids)} games")
    print(f"Test AUC: {roc_auc_score(y_test, probs):.4f}")
    print(f"Test log loss: {log_loss(y_test, probs):.4f}")
    print(f"Test Brier: {brier_score_loss(y_test, probs):.4f}")

    # Sanity check: probability should climb monotonically with time
    # elapsed for a fixed lead, and a big early lead shouldn't yet mean
    # near-certainty.
    sanity_rows = pd.DataFrame([
        # score_diff_home, time_remaining_frac, possession_home, spread_line
        [0, 1.0, 1, 0.0],   # kickoff, pick 'em -> should be ~50%
        [7, 0.5, 1, 0.0],   # home up 7, halftime, pick 'em game
        [7, 0.02, 0, 0.0],  # home up 7, 2 min left, away has ball
        [-14, 0.02, 1, 0.0],  # home down 14, 2 min left -> should be very low
    ], columns=['score_diff_home', 'time_remaining_frac', 'possession_home', 'spread_line'])
    sanity_rows['spread_x_time_remaining'] = sanity_rows['spread_line'] * sanity_rows['time_remaining_frac']
    sanity_rows['score_diff_x_time_elapsed'] = sanity_rows['score_diff_home'] * (1 - sanity_rows['time_remaining_frac'])
    print("\nSanity checks (home win prob):")
    for (_, row), p in zip(sanity_rows.iterrows(), model.predict_proba(sanity_rows[FEATURE_COLS])[:, 1]):
        print(f"  score_diff={row['score_diff_home']:+.0f}, time_left={row['time_remaining_frac']:.0%}, "
              f"home_has_ball={bool(row['possession_home'])} -> {p:.1%}")

    joblib.dump({'model': model, 'feature_cols': FEATURE_COLS}, model_path)
    print(f"\nSaved to {model_path}")
    return model


# --- Pre-game win probability -> spread-equivalent calibration ---
# Fit once against real closing lines, so at serving time we can turn
# our own model's kickoff probability into a "spread_line"-shaped
# number the live model understands.
def fit_prob_to_spread_calibration(seasons=range(2015, 2026)):
    sched = nfl.load_schedules(seasons=seasons).to_pandas()
    sched = sched.dropna(subset=['spread_line', 'home_score', 'away_score'])
    sched['home_win'] = (sched['home_score'] > sched['away_score']).astype(int)

    calib = LogisticRegression()
    calib.fit(sched[['spread_line']], sched['home_win'])
    coef, intercept = calib.coef_[0][0], calib.intercept_[0]

    def prob_to_spread(p):
        p = np.clip(p, 0.02, 0.98)  # avoid extrapolating past the data we validated
        logit = np.log(p / (1 - p))
        return (logit - intercept) / coef

    return prob_to_spread, coef, intercept


if __name__ == "__main__":
    train_live_wp_model()
    prob_to_spread, coef, intercept = fit_prob_to_spread_calibration()
    joblib.dump({'coef': coef, 'intercept': intercept}, 'models/prob_to_spread_calibration.joblib')
    print(f"\nCalibration: spread = (logit(p) - {intercept:.4f}) / {coef:.4f}")
    for p in [0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"  p={p} -> spread={prob_to_spread(p):.2f}")