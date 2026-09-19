import joblib
import numpy as np
import pandas as pd

ESPN_ABBR_MAP = {'WAS': 'WSH', 'LA': 'LAR'}


def _to_espn_abbr(team):
    return ESPN_ABBR_MAP.get(team, team)


_live_model_bundle = None
_calibration_bundle = None


def _load_live_model(model_path='models/live_wp_model.joblib'):
    global _live_model_bundle
    if _live_model_bundle is None:
        _live_model_bundle = joblib.load(model_path)
    return _live_model_bundle


def _load_calibration(path='models/prob_to_spread_calibration.joblib'):
    global _calibration_bundle
    if _calibration_bundle is None:
        _calibration_bundle = joblib.load(path)
    return _calibration_bundle


def pregame_prob_to_spread(p, calibration=None):
    if calibration is None:
        calibration = _load_calibration()
    coef, intercept = calibration['coef'], calibration['intercept']
    p = np.clip(p, 0.02, 0.98)
    logit = np.log(p / (1 - p))
    return (logit - intercept) / coef


def _clock_to_seconds(clock_str):
    try:
        mins, secs = clock_str.split(':')
        return int(mins) * 60 + int(secs)
    except (ValueError, AttributeError, TypeError):
        return 0


def game_seconds_remaining(quarter, clock_str):
    """
    Overtime (quarter 5+) is treated as 0 seconds of regulation
    remaining - the model then leans entirely on score_diff and
    possession, since it was never trained on true sudden-death
    dynamics and shouldn't be trusted to extrapolate into them.
    """
    if quarter is None or quarter > 4:
        return 0
    clock_seconds = _clock_to_seconds(clock_str)
    quarters_remaining_after_this = max(0, 4 - quarter)
    return quarters_remaining_after_this * 900 + clock_seconds


def get_live_win_probability(home_team, away_team, live, pregame_home_win_prob):
    """
    live: the dict from get_live_game_data(), passed in rather than
    fetched again here, so a caller that already fetched it for the
    score display can reuse it instead of hitting ESPN twice.
    """
    if not live or live.get('state') not in ('in', 'post'):
        return pregame_home_win_prob

    if live['state'] == 'post':
        home_score, away_score = int(live['home_score']), int(live['away_score'])
        if home_score > away_score:
            return 1.0
        elif away_score > home_score:
            return 0.0
        else:
            return pregame_home_win_prob

    bundle = _load_live_model()
    model, feature_cols = bundle['model'], bundle['feature_cols']

    spread_line = pregame_prob_to_spread(pregame_home_win_prob)
    score_diff_home = int(live['home_score']) - int(live['away_score'])
    seconds_remaining = game_seconds_remaining(live.get('quarter'), live.get('clock'))
    time_remaining_frac = seconds_remaining / 3600.0
    possession_home = 1 if live.get('possession_team') == _to_espn_abbr(home_team) else 0

    row = {
        'score_diff_home': score_diff_home,
        'time_remaining_frac': time_remaining_frac,
        'possession_home': possession_home,
        'spread_line': spread_line,
        'spread_x_time_remaining': spread_line * time_remaining_frac,
        'score_diff_x_time_elapsed': score_diff_home * (1 - time_remaining_frac),
    }
    X = pd.DataFrame([row])[feature_cols]
    return model.predict_proba(X)[0][1]

if __name__ == "__main__":
    # Mock live-data shapes exactly as get_live_game_data() returns them,
    # to sanity-check the wiring without needing a live game right now.
    cases = [
        ("home leading big, ESPN abbr mismatch (WAS)",
         "WAS", "PHI",
         {'available': True, 'state': 'in', 'quarter': 4, 'clock': '2:00',
          'home_score': '21', 'away_score': '7', 'possession_team': 'WSH',
          'down_distance': '2nd & 8'},
         0.55),
        ("away leading late, home has ball",
         "KC", "DEN",
         {'available': True, 'state': 'in', 'quarter': 4, 'clock': '0:45',
          'home_score': '17', 'away_score': '20', 'possession_team': 'KC'},
         0.70),
        ("game just finished, home lost",
         "MIA", "LV",
         {'available': False, 'state': 'post', 'home_score': '13', 'away_score': '27'},
         0.72),
        ("pre-game, no live data yet",
         "SEA", "NE",
         {'available': False, 'state': 'pre'},
         0.75),
    ]
    for label, home, away, live, pregame_p in cases:
        p = get_live_win_probability(home, away, live, pregame_p)
        print(f"{label}: pregame={pregame_p:.0%} -> live={p:.1%}")