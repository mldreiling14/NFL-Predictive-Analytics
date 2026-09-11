import pandas as pd
import sqlite3
import nflreadpy as nfl

def safe_load_pfr_advstats(seasons, stat_type):
    """
    Pulls PFR advanced stats season-by-season, skipping any season
    whose data hasn't been published yet (common early in a new
    season - the underlying files simply don't exist yet on
    nflverse's servers). Automatically picks up newly available
    seasons later without needing any code changes.
    """
    frames = []
    for s in seasons:
        try:
            df = nfl.load_pfr_advstats(seasons=[s], stat_type=stat_type).to_pandas()
            frames.append(df)
        except Exception as e:
            print(f"Skipping PFR {stat_type} stats for {s}: {e}")
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()
    
def build_snapshots(db_path="data/nfl.db", seasons=range(2015, 2026)):
    """
    Builds every 'current state' table needed for live predictions:
    each team's current form, offense/defense ratings, QB/RB/WR-TE
    stats, coach, defense-allowed, O-line, and primary WR/CB/pass-
    rusher for matchup and report purposes.

    Unlike the training features (which use shift(1) to exclude each
    row's own game), these use every real completed game INCLUDING
    the most recent one, since that IS each team's current state
    entering their next (not-yet-played) game.

    Returns a dict of DataFrames, one per snapshot type.
    """
    conn = sqlite3.connect(db_path)
    df_full = pd.read_sql_query("SELECT * FROM games_with_features", conn)
    conn.close()
    df_full['home_win'] = (df_full['home_score'] > df_full['away_score']).astype(int)

    game_dates = df_full[['game_id', 'gameday']].drop_duplicates()

    player_stats = nfl.load_player_stats(seasons=list(seasons)).to_pandas()
    snaps_full = nfl.load_snap_counts(seasons=list(seasons)).to_pandas()
    players = nfl.load_players().to_pandas()
    crosswalk = players[['gsis_id', 'pfr_id']].dropna().rename(
        columns={'gsis_id': 'player_id', 'pfr_id': 'pfr_player_id'})
    physical = players[['pfr_id', 'height', 'weight']].dropna().rename(columns={'pfr_id': 'pfr_player_id'})

    pfr_seasons = [s for s in seasons if s >= 2018]
    adv_def_full = safe_load_pfr_advstats(pfr_seasons, 'def')
    adv_def_full = adv_def_full[['game_id', 'season', 'week', 'team', 'pfr_player_id',
                                  'def_completion_pct', 'def_passer_rating_allowed']]

    home = df_full[['game_id', 'gameday', 'home_team_std', 'home_win', 'home_score', 'away_score']].rename(
        columns={'home_team_std': 'team'})
    home['win'] = home['home_win']
    home['point_diff'] = home['home_score'] - home['away_score']
    away = df_full[['game_id', 'gameday', 'away_team_std', 'home_win', 'home_score', 'away_score']].rename(
        columns={'away_team_std': 'team'})
    away['win'] = 1 - away['home_win']
    away['point_diff'] = away['away_score'] - away['home_score']
    team_games = pd.concat([home, away], ignore_index=True).sort_values(['team', 'gameday']).reset_index(drop=True)

    current_form = (
        team_games.groupby('team').apply(lambda g: g.tail(5)[['win', 'point_diff']].mean())
        .reset_index().rename(columns={'win': 'current_recent_form', 'point_diff': 'current_recent_point_diff'})
    )

    elo_games = df_full[['game_id', 'season', 'gameday', 'home_team_std', 'away_team_std',
                          'home_score', 'away_score']].copy()
    elo_games = elo_games.sort_values(['season', 'gameday']).reset_index(drop=True)
    all_teams = set(elo_games['home_team_std']).union(set(elo_games['away_team_std']))
    league_avg_points = elo_games[['home_score', 'away_score']].values.mean()

    off_rating = {team: 0.0 for team in all_teams}
    def_rating = {team: 0.0 for team in all_teams}
    k_off_def, revert_fraction = 0.05, 1/3
    current_season = None

    for _, row in elo_games.iterrows():
        if current_season is not None and row['season'] != current_season:
            for team in off_rating:
                off_rating[team] *= (1 - revert_fraction)
                def_rating[team] *= (1 - revert_fraction)
        current_season = row['season']

        h, a = row['home_team_std'], row['away_team_std']
        expected_home_score = league_avg_points + off_rating[h] - def_rating[a]
        expected_away_score = league_avg_points + off_rating[a] - def_rating[h]

        off_rating[h] += k_off_def * (row['home_score'] - expected_home_score)
        def_rating[a] -= k_off_def * (row['home_score'] - expected_home_score)
        off_rating[a] += k_off_def * (row['away_score'] - expected_away_score)
        def_rating[h] -= k_off_def * (row['away_score'] - expected_away_score)

    for team in off_rating:
        off_rating[team] *= (1 - revert_fraction)
        def_rating[team] *= (1 - revert_fraction)

    current_off_def = pd.DataFrame({
        'team': list(off_rating.keys()),
        'current_off_rating': list(off_rating.values()),
        'current_def_rating': [def_rating[t] for t in off_rating.keys()]
    })

    depth = nfl.load_depth_charts(seasons=[max(seasons)]).to_pandas()
    qb_depth = depth[(depth['pos_abb'] == 'QB') & (depth['pos_rank'] == 1)].copy()
    qb_depth['dt'] = pd.to_datetime(qb_depth['dt'])
    current_qb_starter = qb_depth.loc[qb_depth.groupby('team')['dt'].idxmax()][['team', 'gsis_id']].rename(
        columns={'gsis_id': 'player_id'})

    qb_stats = player_stats[(player_stats['position'] == 'QB') & (player_stats['attempts'] > 0)].copy()
    qb_stats = qb_stats[['player_id', 'game_id', 'passing_yards', 'passing_tds',
                         'passing_interceptions', 'passing_epa']]
    qb_stats = qb_stats.merge(game_dates, on='game_id', how='left').sort_values(['player_id', 'gameday']).reset_index(drop=True)
    qb_stats_raw = qb_stats.copy()
    for col in ['passing_yards', 'passing_tds', 'passing_interceptions', 'passing_epa']:
        qb_stats[f'recent_{col}'] = qb_stats.groupby('player_id')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean())

    each_qb_current = qb_stats.loc[qb_stats.groupby('player_id')['gameday'].idxmax()][
        ['player_id', 'recent_passing_yards', 'recent_passing_tds', 'recent_passing_interceptions', 'recent_passing_epa']]

    current_qb = current_qb_starter.merge(each_qb_current, on='player_id', how='left')

    rb_stats = player_stats[player_stats['position'] == 'RB'].copy()
    rb_stats = rb_stats[['player_id', 'game_id', 'team', 'rushing_yards', 'rushing_epa', 'receiving_yards']]
    rb_stats = rb_stats.merge(crosswalk, on='player_id', how='left')
    rb_stats = rb_stats.merge(
        snaps_full[snaps_full['position'] == 'RB'][['pfr_player_id', 'game_id', 'offense_pct']],
        on=['pfr_player_id', 'game_id'], how='left')
    rb_stats = rb_stats.merge(game_dates, on='game_id', how='left').sort_values(['player_id', 'gameday']).reset_index(drop=True)
    rb_stats['offense_pct'] = rb_stats.groupby('player_id')['offense_pct'].ffill()
    rb_stats['w_rush_yards'] = rb_stats['rushing_yards'] * rb_stats['offense_pct']
    rb_stats['w_rush_epa'] = rb_stats['rushing_epa'] * rb_stats['offense_pct']
    rb_stats['w_rec_yards'] = rb_stats['receiving_yards'] * rb_stats['offense_pct']

    rb_team_game = rb_stats.groupby(['team', 'game_id', 'gameday'], as_index=False).agg(
        t_rush_yards=('w_rush_yards', 'sum'), t_rush_epa=('w_rush_epa', 'sum'), t_rec_yards=('w_rec_yards', 'sum'))
    rb_team_game = rb_team_game.sort_values(['team', 'gameday']).reset_index(drop=True)
    for col in ['t_rush_yards', 't_rush_epa', 't_rec_yards']:
        rb_team_game[f'recent_{col}'] = rb_team_game.groupby('team')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean())
    current_rb = rb_team_game.loc[rb_team_game.groupby('team')['gameday'].idxmax()][
        ['team', 'recent_t_rush_yards', 'recent_t_rush_epa', 'recent_t_rec_yards']]

    rb_depth = depth[(depth['pos_abb'] == 'RB') & (depth['pos_rank'] == 1)].copy()
    rb_depth['dt'] = pd.to_datetime(rb_depth['dt'])
    current_rb_starter = rb_depth.loc[rb_depth.groupby('team')['dt'].idxmax()][['team', 'gsis_id']].rename(
        columns={'gsis_id': 'player_id'})

    wrte_stats = player_stats[player_stats['position'].isin(['WR', 'TE'])].copy()
    wrte_stats = wrte_stats[['player_id', 'game_id', 'team', 'targets', 'receiving_yards', 'receiving_epa']]
    wrte_stats = wrte_stats.merge(crosswalk, on='player_id', how='left')
    wrte_stats = wrte_stats.merge(
        snaps_full[snaps_full['position'].isin(['WR', 'TE'])][['pfr_player_id', 'game_id', 'offense_pct']],
        on=['pfr_player_id', 'game_id'], how='left')
    wrte_stats = wrte_stats.merge(game_dates, on='game_id', how='left').sort_values(['player_id', 'gameday']).reset_index(drop=True)
    wrte_stats['offense_pct'] = wrte_stats.groupby('player_id')['offense_pct'].ffill()
    wrte_stats['w_rec_yards'] = wrte_stats['receiving_yards'] * wrte_stats['offense_pct']
    wrte_stats['w_rec_epa'] = wrte_stats['receiving_epa'] * wrte_stats['offense_pct']
    wrte_stats['w_targets'] = wrte_stats['targets'] * wrte_stats['offense_pct']

    wrte_team_game = wrte_stats.groupby(['team', 'game_id', 'gameday'], as_index=False).agg(
        t_rec_yards=('w_rec_yards', 'sum'), t_rec_epa=('w_rec_epa', 'sum'), t_targets=('w_targets', 'sum'))
    wrte_team_game = wrte_team_game.sort_values(['team', 'gameday']).reset_index(drop=True)
    for col in ['t_rec_yards', 't_rec_epa', 't_targets']:
        wrte_team_game[f'recent_{col}'] = wrte_team_game.groupby('team')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean())
    current_wrte = wrte_team_game.loc[wrte_team_game.groupby('team')['gameday'].idxmax()][
        ['team', 'recent_t_rec_yards', 'recent_t_rec_epa', 'recent_t_targets']]

    hc = df_full[['home_team_std', 'home_coach', 'gameday']].rename(columns={'home_team_std': 'team', 'home_coach': 'coach'})
    ac = df_full[['away_team_std', 'away_coach', 'gameday']].rename(columns={'away_team_std': 'team', 'away_coach': 'coach'})
    all_coach = pd.concat([hc, ac], ignore_index=True)
    current_coach = all_coach.loc[all_coach.groupby('team')['gameday'].idxmax()][['team', 'coach']]

    ts = nfl.load_team_stats(seasons=list(seasons)).to_pandas()
    ts = ts[['game_id', 'team', 'opponent_team', 'passing_epa', 'rushing_epa', 'passing_yards',
             'rushing_yards', 'def_interceptions', 'fumble_recovery_opp', 'attempts', 'sacks_suffered']]
    opp = ts[['game_id', 'team', 'passing_epa', 'rushing_epa', 'passing_yards', 'rushing_yards']].rename(
        columns={'team': 'opponent_team', 'passing_epa': 'opp_pass_epa', 'rushing_epa': 'opp_rush_epa',
                 'passing_yards': 'opp_pass_yds', 'rushing_yards': 'opp_rush_yds'})
    ts = ts.merge(opp, on=['game_id', 'opponent_team'], how='left')
    ts['epa_allowed'] = ts['opp_pass_epa'] + ts['opp_rush_epa']
    ts['yards_allowed'] = ts['opp_pass_yds'] + ts['opp_rush_yds']
    ts['takeaways'] = ts['def_interceptions'] + ts['fumble_recovery_opp']
    ts['sack_rate'] = ts['sacks_suffered'] / (ts['attempts'] + ts['sacks_suffered'])
    ts = ts.merge(game_dates, on='game_id', how='left').sort_values(['team', 'gameday']).reset_index(drop=True)
    for col in ['epa_allowed', 'yards_allowed', 'takeaways', 'sack_rate']:
        ts[f'recent_{col}'] = ts.groupby('team')[col].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    current_team_stats = ts.loc[ts.groupby('team')['gameday'].idxmax()][
        ['team', 'recent_epa_allowed', 'recent_yards_allowed', 'recent_takeaways', 'recent_sack_rate']]

    press = safe_load_pfr_advstats(pfr_seasons, 'pass')
    press = press[['game_id', 'team', 'times_pressured_pct']]
    press = press.merge(game_dates, on='game_id', how='left').sort_values(['team', 'gameday']).reset_index(drop=True)
    press['recent_pressure_pct'] = press.groupby('team')['times_pressured_pct'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean())
    current_pressure = press.loc[press.groupby('team')['gameday'].idxmax()][['team', 'recent_pressure_pct']]

    wr_depth = depth[(depth['pos_abb'] == 'WR') & (depth['pos_rank'] == 1)].copy()
    wr_depth['dt'] = pd.to_datetime(wr_depth['dt'])
    current_wr_starter = wr_depth.loc[wr_depth.groupby('team')['dt'].idxmax()][['team', 'gsis_id']].rename(
        columns={'gsis_id': 'player_id'})
    current_wr_starter = current_wr_starter.merge(crosswalk, on='player_id', how='left')
    current_primary_wr = current_wr_starter[['team', 'pfr_player_id']].merge(physical, on='pfr_player_id', how='left')

    cb_depth = depth[(depth['pos_abb'].isin(['LCB', 'RCB'])) & (depth['pos_rank'] == 1)].copy()
    cb_depth['dt'] = pd.to_datetime(cb_depth['dt'])
    current_cb_starters = cb_depth.loc[cb_depth.groupby(['team', 'pos_abb'])['dt'].idxmax()][
        ['team', 'gsis_id', 'pos_abb']].rename(columns={'gsis_id': 'player_id'})
    current_cb_starters = current_cb_starters.merge(crosswalk, on='player_id', how='left')

    cb_stats_all = adv_def_full[['game_id', 'pfr_player_id', 'def_completion_pct', 'def_passer_rating_allowed']].copy()
    cb_stats_all = cb_stats_all.merge(game_dates, on='game_id', how='left').sort_values(
        ['pfr_player_id', 'gameday']).reset_index(drop=True)
    for col in ['def_completion_pct', 'def_passer_rating_allowed']:
        cb_stats_all[f'recent_{col}'] = cb_stats_all.groupby('pfr_player_id')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean())
    each_cb_current = cb_stats_all.loc[cb_stats_all.groupby('pfr_player_id')['gameday'].idxmax()][
        ['pfr_player_id', 'recent_def_completion_pct', 'recent_def_passer_rating_allowed']]

    current_cb_starters = current_cb_starters.merge(each_cb_current, on='pfr_player_id', how='left')

    current_primary_cb = current_cb_starters.loc[
        current_cb_starters.groupby('team')['recent_def_completion_pct'].idxmin()
    ][['team', 'pfr_player_id', 'recent_def_completion_pct', 'recent_def_passer_rating_allowed']]
    current_primary_cb = current_primary_cb.merge(physical, on='pfr_player_id', how='left')

    player_names = players[['gsis_id', 'display_name']].rename(columns={'gsis_id': 'player_id'})
    player_names_pfr = players[['pfr_id', 'display_name']].rename(columns={'pfr_id': 'pfr_player_id'})

    current_qb = current_qb.merge(player_names, on='player_id', how='left')
    current_rb_starter = current_rb_starter.merge(player_names, on='player_id', how='left')
    current_primary_wr = current_primary_wr.merge(player_names_pfr, on='pfr_player_id', how='left')
    current_primary_cb = current_primary_cb.merge(player_names_pfr, on='pfr_player_id', how='left')

    def_positions = ['DE', 'DT', 'DL', 'NT', 'LB', 'ILB', 'OLB', 'MLB', 'SLB', 'WLB']
    pass_rush_stats = player_stats[player_stats['position'].isin(def_positions)][
        ['player_id', 'game_id', 'team', 'def_sacks', 'def_qb_hits']].copy()
    pass_rush_stats['pressure_score'] = pass_rush_stats['def_sacks'] + (pass_rush_stats['def_qb_hits'] * 0.5)
    pass_rush_stats = pass_rush_stats.merge(game_dates, on='game_id', how='left')
    pass_rush_stats = pass_rush_stats.sort_values(['player_id', 'gameday']).reset_index(drop=True)

    pass_rush_stats['recent_pressure_score'] = (
        pass_rush_stats.groupby('player_id')['pressure_score']
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    )

    each_rusher_current = pass_rush_stats.loc[pass_rush_stats.groupby('player_id')['gameday'].idxmax()][
        ['player_id', 'team', 'recent_pressure_score']]

    threshold = each_rusher_current['recent_pressure_score'].quantile(0.90)
    each_rusher_current['is_star_rusher'] = (each_rusher_current['recent_pressure_score'] >= threshold).astype(int)

    current_pass_rusher = each_rusher_current.loc[
        each_rusher_current.groupby('team')['recent_pressure_score'].idxmax()
    ][['team', 'player_id', 'recent_pressure_score', 'is_star_rusher']]
    current_pass_rusher = current_pass_rusher.merge(player_names, on='player_id', how='left')

    return {
        'df_full': df_full,
        'current_form': current_form,
        'current_off_def': current_off_def,
        'current_qb': current_qb,
        'current_rb': current_rb,
        'current_rb_starter': current_rb_starter,
        'current_wrte': current_wrte,
        'current_coach': current_coach,
        'current_team_stats': current_team_stats,
        'current_pressure': current_pressure,
        'current_primary_wr': current_primary_wr,
        'current_primary_cb': current_primary_cb,
        'current_pass_rusher': current_pass_rusher,
        'qb_stats_raw': qb_stats_raw,
    }