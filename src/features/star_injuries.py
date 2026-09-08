import pandas as pd
from .common import standardize_team_codes


def add_star_rb_injury_feature(df_full, player_stats, snaps_full, crosswalk, injuries, window=5):
    """
    Adds a flag for whether a team's primary RB (by recent snap share)
    is BOTH a genuine star (top 10% league-wide by rolling fantasy PPR)
    AND carrying an injury designation. Real-world check: home win
    rate 48.7% vs 54.8% overall when out (n=76). Kept despite a small
    net accuracy cost in isolated testing - see FEATURES.md.
    """

    rb_fantasy = player_stats[player_stats['position'] == 'RB'][
        ['player_id', 'game_id', 'season', 'week', 'fantasy_points_ppr']].copy()
    rb_fantasy = rb_fantasy.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
    rb_fantasy['recent_fantasy_ppr'] = (
        rb_fantasy.groupby('player_id')['fantasy_points_ppr']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    rb_fantasy['position_percentile'] = (
        rb_fantasy.groupby('game_id')['recent_fantasy_ppr'].rank(pct=True)
    )
    rb_fantasy['is_star'] = (rb_fantasy['position_percentile'] >= 0.90).astype(int)
    star_lookup = rb_fantasy[['player_id', 'game_id', 'is_star']]

    rb_stats_local = player_stats[player_stats['position'] == 'RB'][
        ['player_id', 'game_id', 'season', 'week', 'team']].copy()
    rb_stats_local = rb_stats_local.merge(crosswalk, on='player_id', how='left')
    rb_stats_local = rb_stats_local.merge(
        snaps_full[snaps_full['position'] == 'RB'][['pfr_player_id', 'game_id', 'offense_pct']],
        on=['pfr_player_id', 'game_id'], how='left')

    rb_sorted = rb_stats_local.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
    rb_sorted['recent_snap_pct'] = (
        rb_sorted.groupby(['player_id', 'season'])['offense_pct']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    primary_rb = (
        rb_sorted.sort_values('recent_snap_pct', ascending=False)
        .groupby(['team', 'game_id'], as_index=False)
        .first()[['team', 'game_id', 'season', 'week', 'player_id']]
    )

    primary_rb = primary_rb.merge(star_lookup, on=['player_id', 'game_id'], how='left')
    primary_rb['is_star'] = primary_rb['is_star'].fillna(0)

    rb_injuries = injuries[injuries['position'] == 'RB'].copy()
    rb_injuries['injury_flag'] = rb_injuries['report_status'].isin(
        ['Out', 'Doubtful', 'Questionable']).astype(int)
    rb_injuries = rb_injuries.rename(columns={'gsis_id': 'player_id'})

    primary_rb = primary_rb.merge(
        rb_injuries[['season', 'week', 'team', 'player_id', 'injury_flag']],
        on=['season', 'week', 'team', 'player_id'], how='left')
    primary_rb['injury_flag'] = primary_rb['injury_flag'].fillna(0).astype(int)
    primary_rb['star_rb_injured'] = (
        (primary_rb['is_star'] == 1) & (primary_rb['injury_flag'] == 1)
    ).astype(int)

    df_full = standardize_team_codes(df_full)

    home_star_rb = primary_rb[['season', 'week', 'team', 'star_rb_injured']].rename(
        columns={'team': 'home_team_std', 'star_rb_injured': 'home_star_rb_injured'})
    away_star_rb = primary_rb[['season', 'week', 'team', 'star_rb_injured']].rename(
        columns={'team': 'away_team_std', 'star_rb_injured': 'away_star_rb_injured'})

    df_full = df_full.drop(columns=['home_star_rb_injured', 'away_star_rb_injured'], errors='ignore')
    df_full = df_full.merge(home_star_rb, on=['season', 'week', 'home_team_std'], how='left')
    df_full = df_full.merge(away_star_rb, on=['season', 'week', 'away_team_std'], how='left')
    df_full['home_star_rb_injured'] = df_full['home_star_rb_injured'].fillna(0).astype(int)
    df_full['away_star_rb_injured'] = df_full['away_star_rb_injured'].fillna(0).astype(int)

    return df_full


def add_star_wr_injury_feature(df_full, player_stats, snaps_full, crosswalk, injuries, window=5):
    """
    Adds a flag for whether a team's primary WR (by recent snap share)
    is BOTH a genuine star (top 5% league-wide by rolling targets) AND
    carrying a serious injury designation (Out/Doubtful only -
    Questionable excluded, since those players often still play).

    BUG HISTORY: an earlier version identified "primary WR"/"star"
    status using only games where the player has a real stat row - an
    injured player generates NO row for games missed, so they became
    invisible to their own checks at exactly the moment they were out,
    producing a counterintuitive result (higher win rate when the
    "star" was "out"). Fixed using pd.merge_asof to project each
    player's last known snap share AND star status forward across
    every game their team plays, so an injured star's real role still
    "shows up" until the injury check runs.

    Real-world check after fix: home win rate ~38% vs ~54.8% overall
    when out/doubtful (n=21, small sample). Confirmed to add real
    accuracy value in isolated testing (67.4% -> 67.7%).
    """

    game_dates = df_full[['game_id', 'gameday']].drop_duplicates()
    game_dates['gameday'] = pd.to_datetime(game_dates['gameday'])

    df_full = standardize_team_codes(df_full)

    team_game_dates = pd.concat([
        df_full[['home_team_std', 'game_id', 'gameday']].rename(columns={'home_team_std': 'team'}),
        df_full[['away_team_std', 'game_id', 'gameday']].rename(columns={'away_team_std': 'team'})
    ], ignore_index=True)
    team_game_dates['gameday'] = pd.to_datetime(team_game_dates['gameday'])
    team_game_dates = team_game_dates.sort_values('gameday').reset_index(drop=True)

    wr_stats = player_stats[player_stats['position'] == 'WR'][
        ['player_id', 'game_id', 'season', 'week', 'team', 'targets']].copy()
    wr_stats = wr_stats.merge(crosswalk, on='player_id', how='left')
    wr_stats = wr_stats.merge(
        snaps_full[snaps_full['position'] == 'WR'][['pfr_player_id', 'game_id', 'offense_pct']],
        on=['pfr_player_id', 'game_id'], how='left')
    wr_stats = wr_stats.merge(game_dates, on='game_id', how='left')
    wr_stats = wr_stats.sort_values(['player_id', 'gameday']).reset_index(drop=True)

    wr_stats['recent_snap_pct'] = (
        wr_stats.groupby('player_id')['offense_pct']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    wr_stats['recent_targets'] = (
        wr_stats.groupby('player_id')['targets']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    wr_stats['target_percentile'] = wr_stats.groupby('game_id')['recent_targets'].rank(pct=True)
    wr_stats['is_star_wr'] = (wr_stats['target_percentile'] >= 0.95).astype(int)

    snap_history = wr_stats[['player_id', 'team', 'gameday', 'recent_snap_pct']].dropna(subset=['gameday', 'recent_snap_pct'])
    star_history = wr_stats[['player_id', 'gameday', 'is_star_wr']].dropna(subset=['gameday'])

    snap_projected_frames = []
    for player_id, player_df in snap_history.sort_values('gameday').groupby('player_id'):
        team = player_df['team'].iloc[-1]
        team_games = team_game_dates[team_game_dates['team'] == team].sort_values('gameday')
        projected = pd.merge_asof(
            team_games, player_df[['gameday', 'recent_snap_pct']].sort_values('gameday'),
            on='gameday', direction='backward'
        )
        projected['player_id'] = player_id
        snap_projected_frames.append(projected)
    wr_snap_projected = pd.concat(snap_projected_frames, ignore_index=True).dropna(subset=['recent_snap_pct'])

    star_projected_frames = []
    for player_id, player_df in star_history.sort_values('gameday').groupby('player_id'):
        player_teams = snap_history[snap_history['player_id'] == player_id]['team']
        if len(player_teams) == 0:
            continue
        team = player_teams.iloc[-1]
        team_games = team_game_dates[team_game_dates['team'] == team].sort_values('gameday')
        projected = pd.merge_asof(
            team_games, player_df[['gameday', 'is_star_wr']].sort_values('gameday'),
            on='gameday', direction='backward'
        )
        projected['player_id'] = player_id
        star_projected_frames.append(projected)
    wr_star_projected = pd.concat(star_projected_frames, ignore_index=True).dropna(subset=['is_star_wr'])
    wr_star_projected = wr_star_projected[['player_id', 'game_id', 'is_star_wr']]

    primary_wr = (
        wr_snap_projected.sort_values('recent_snap_pct', ascending=False)
        .groupby(['team', 'game_id'], as_index=False)
        .first()[['team', 'game_id', 'player_id']]
    )
    primary_wr = primary_wr.merge(wr_star_projected, on=['player_id', 'game_id'], how='left')
    primary_wr['is_star_wr'] = primary_wr['is_star_wr'].fillna(0)

    wr_injuries = injuries[injuries['position'] == 'WR'].copy()
    wr_injuries['injury_flag_tight'] = wr_injuries['report_status'].isin(['Out', 'Doubtful']).astype(int)
    wr_injuries = wr_injuries.rename(columns={'gsis_id': 'player_id'})

    primary_wr = primary_wr.merge(df_full[['game_id', 'season', 'week']].drop_duplicates(), on='game_id', how='left')
    primary_wr = primary_wr.merge(
        wr_injuries[['season', 'week', 'team', 'player_id', 'injury_flag_tight']],
        on=['season', 'week', 'team', 'player_id'], how='left')
    primary_wr['injury_flag_tight'] = primary_wr['injury_flag_tight'].fillna(0).astype(int)

    primary_wr['star_wr_injured'] = (
        (primary_wr['is_star_wr'] == 1) & (primary_wr['injury_flag_tight'] == 1)
    ).astype(int)

    home_star_wr = primary_wr[['season', 'week', 'team', 'star_wr_injured']].rename(
        columns={'team': 'home_team_std', 'star_wr_injured': 'home_star_wr_injured'})
    away_star_wr = primary_wr[['season', 'week', 'team', 'star_wr_injured']].rename(
        columns={'team': 'away_team_std', 'star_wr_injured': 'away_star_wr_injured'})

    df_full = df_full.drop(columns=['home_star_wr_injured', 'away_star_wr_injured'], errors='ignore')
    df_full = df_full.merge(home_star_wr, on=['season', 'week', 'home_team_std'], how='left')
    df_full = df_full.merge(away_star_wr, on=['season', 'week', 'away_team_std'], how='left')
    df_full['home_star_wr_injured'] = df_full['home_star_wr_injured'].fillna(0).astype(int)
    df_full['away_star_wr_injured'] = df_full['away_star_wr_injured'].fillna(0).astype(int)

    return df_full