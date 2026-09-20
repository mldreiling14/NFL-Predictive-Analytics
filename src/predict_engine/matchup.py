import pandas as pd
import nflreadpy as nfl

from .injuries import get_injured_player_ids


def _coach_h2h(df_full, home_coach, away_coach):
    meetings = df_full[
        ((df_full['home_coach'] == home_coach) & (df_full['away_coach'] == away_coach)) |
        ((df_full['home_coach'] == away_coach) & (df_full['away_coach'] == home_coach))
    ]
    if len(meetings) == 0:
        return 0.5, 0
    wins = (
        ((meetings['home_coach'] == home_coach) & (meetings['home_win'] == 1)) |
        ((meetings['away_coach'] == home_coach) & (meetings['home_win'] == 0))
    ).sum()
    return wins / len(meetings), len(meetings)


def build_matchup_features(home_team, away_team, snapshots, home_rest, away_rest, div_game, spread_line, injured_ids):
    """Assembles one feature row for a specific upcoming matchup from the snapshot tables."""
    s = snapshots
    row = {}

    row['home_recent_form'] = s['current_form'].loc[s['current_form']['team'] == home_team, 'current_recent_form'].values[0]
    row['away_recent_form'] = s['current_form'].loc[s['current_form']['team'] == away_team, 'current_recent_form'].values[0]
    row['home_recent_point_diff'] = s['current_form'].loc[s['current_form']['team'] == home_team, 'current_recent_point_diff'].values[0]
    row['away_recent_point_diff'] = s['current_form'].loc[s['current_form']['team'] == away_team, 'current_recent_point_diff'].values[0]

    for side, team in [('home', home_team), ('away', away_team)]:
        qb = s['current_qb'][s['current_qb']['team'] == team]
        row[f'{side}_qb_recent_yards'] = qb['recent_passing_yards'].values[0]
        row[f'{side}_qb_recent_tds'] = qb['recent_passing_tds'].values[0]
        row[f'{side}_qb_recent_ints'] = qb['recent_passing_interceptions'].values[0]
        row[f'{side}_qb_recent_epa'] = qb['recent_passing_epa'].values[0]

        rb = s['current_rb'][s['current_rb']['team'] == team]
        row[f'{side}_rb_recent_rush_yards'] = rb['recent_t_rush_yards'].values[0]
        row[f'{side}_rb_recent_rush_epa'] = rb['recent_t_rush_epa'].values[0]
        row[f'{side}_rb_recent_rec_yards'] = rb['recent_t_rec_yards'].values[0]

        wrte = s['current_wrte'][s['current_wrte']['team'] == team]
        row[f'{side}_wrte_recent_rec_yards'] = wrte['recent_t_rec_yards'].values[0]
        row[f'{side}_wrte_recent_rec_epa'] = wrte['recent_t_rec_epa'].values[0]
        row[f'{side}_wrte_recent_targets'] = wrte['recent_t_targets'].values[0]

        qb_player_id = qb['player_id'].values[0] if len(qb) > 0 else None
        row[f'{side}_qb_injury_flag'] = int(qb_player_id in injured_ids) if qb_player_id else 0

        rb_starter = s['current_rb_starter'][s['current_rb_starter']['team'] == team]
        rb_player_id = rb_starter['player_id'].values[0] if len(rb_starter) > 0 else None
        row[f'{side}_rb_injury_flag'] = int(rb_player_id in injured_ids) if rb_player_id else 0

        wr = s['current_primary_wr'][s['current_primary_wr']['team'] == team]
        wr_player_id = wr['player_id'].values[0] if len(wr) > 0 else None
        te = s['current_te_starter'][s['current_te_starter']['team'] == team]
        te_player_id = te['player_id'].values[0] if len(te) > 0 else None
        wrte_injured = (bool(wr_player_id) and wr_player_id in injured_ids) or \
                       (bool(te_player_id) and te_player_id in injured_ids)
        row[f'{side}_wrte_injury_flag'] = int(wrte_injured)

        # Deferred: needs a league-wide percentile ranking (top 10%/5%
        # of ALL RBs/WRs that week), not just the two teams in this
        # matchup - see FEATURES.md discussion.
        row[f'{side}_star_rb_injured'] = 0
        row[f'{side}_star_wr_injured'] = 0

        ts = s['current_team_stats'][s['current_team_stats']['team'] == team]
        row[f'{side}_epa_allowed_recent'] = ts['recent_epa_allowed'].values[0]
        row[f'{side}_yards_allowed_recent'] = ts['recent_yards_allowed'].values[0]
        row[f'{side}_takeaways_recent'] = ts['recent_takeaways'].values[0]
        row[f'{side}_sack_rate_recent'] = ts['recent_sack_rate'].values[0]

        p = s['current_pressure'][s['current_pressure']['team'] == team]
        row[f'{side}_pressure_pct_recent'] = p['recent_pressure_pct'].values[0] if len(p) > 0 else None

    home_coach = s['current_coach'].loc[s['current_coach']['team'] == home_team, 'coach'].values[0]
    away_coach = s['current_coach'].loc[s['current_coach']['team'] == away_team, 'coach'].values[0]
    row['home_coach_h2h_wins'], row['h2h_games_played'] = _coach_h2h(s['df_full'], home_coach, away_coach)

    home_od = s['current_off_def'][s['current_off_def']['team'] == home_team]
    away_od = s['current_off_def'][s['current_off_def']['team'] == away_team]
    row['home_off_rating_pre'] = home_od['current_off_rating'].values[0]
    row['home_def_rating_pre'] = home_od['current_def_rating'].values[0]
    row['away_off_rating_pre'] = away_od['current_off_rating'].values[0]
    row['away_def_rating_pre'] = away_od['current_def_rating'].values[0]
    row['rest_advantage'] = home_rest - away_rest

    home_wr = s['current_primary_wr'][s['current_primary_wr']['team'] == home_team]
    away_cb = s['current_primary_cb'][s['current_primary_cb']['team'] == away_team]
    row['home_wr_height_advantage'] = home_wr['height'].values[0] - away_cb['height'].values[0]
    row['home_wr_weight_advantage'] = home_wr['weight'].values[0] - away_cb['weight'].values[0]
    row['home_opp_cb_completion_allowed'] = away_cb['recent_def_completion_pct'].values[0]
    row['home_opp_cb_rating_allowed'] = away_cb['recent_def_passer_rating_allowed'].values[0]

    away_wr = s['current_primary_wr'][s['current_primary_wr']['team'] == away_team]
    home_cb = s['current_primary_cb'][s['current_primary_cb']['team'] == home_team]
    row['away_wr_height_advantage'] = away_wr['height'].values[0] - home_cb['height'].values[0]
    row['away_wr_weight_advantage'] = away_wr['weight'].values[0] - home_cb['weight'].values[0]
    row['away_opp_cb_completion_allowed'] = home_cb['recent_def_completion_pct'].values[0]
    row['away_opp_cb_rating_allowed'] = home_cb['recent_def_passer_rating_allowed'].values[0]

    row['div_game'] = div_game
    row['spread_line'] = spread_line

    return pd.DataFrame([row])


def predict_matchup(home_team, away_team, snapshots, model_bundle, home_rest, away_rest, div_game, spread_line, injured_ids):
    """Returns home team win probability (float 0-1) for one matchup."""
    feature_row = build_matchup_features(home_team, away_team, snapshots, home_rest, away_rest, div_game, spread_line, injured_ids)
    feature_cols = model_bundle['feature_cols']
    X = feature_row[feature_cols]
    X_imputed = pd.DataFrame(model_bundle['imputer'].transform(X), columns=feature_cols)
    return model_bundle['model'].predict_proba(X_imputed)[0][1]


DETAIL_COLS = [
    'home_recent_form', 'away_recent_form',
    'home_off_rating_pre', 'home_def_rating_pre',
    'away_off_rating_pre', 'away_def_rating_pre',
    'home_coach_h2h_wins', 'h2h_games_played',
    'home_qb_injury_flag', 'away_qb_injury_flag',
    'home_rb_injury_flag', 'away_rb_injury_flag',
    'home_wrte_injury_flag', 'away_wrte_injury_flag',
    'home_star_rb_injured', 'away_star_rb_injured',
    'home_star_wr_injured', 'away_star_wr_injured',
    'rest_advantage', 'spread_line',
]


def predict_week(season, week, snapshots, model_bundle):
    """Returns a DataFrame of predictions for every game in a given season/week,
    including the underlying feature values needed for a detailed game view."""
    sched = nfl.load_schedules(seasons=[season]).to_pandas()
    week_games = sched[sched['week'] == week]
    injured_ids = get_injured_player_ids(season, week)

    results = []
    for _, g in week_games.iterrows():
        try:
            feature_row = build_matchup_features(
                g['home_team'], g['away_team'], snapshots,
                g['home_rest'], g['away_rest'], g['div_game'], g['spread_line'], injured_ids
            )
            feature_cols = model_bundle['feature_cols']
            X = feature_row[feature_cols]
            X_imputed = pd.DataFrame(model_bundle['imputer'].transform(X), columns=feature_cols)
            prob = model_bundle['model'].predict_proba(X_imputed)[0][1]

            row_dict = {
                'game_id': g['game_id'], 'home_team': g['home_team'], 'away_team': g['away_team'],
                'gameday': g['gameday'], 'home_win_prob': prob, 'away_win_prob': 1 - prob
            }
            for col in DETAIL_COLS:
                if col in feature_row.columns:
                    row_dict[col] = feature_row[col].values[0]

            results.append(row_dict)
        except (IndexError, KeyError) as e:
            print(f"Could not predict {g['game_id']}: missing data ({e})")

    return pd.DataFrame(results)