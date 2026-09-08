from .common import standardize_team_codes


def add_injury_features(df_full, injuries, rb_stats, wrte_stats, window=5):
    """
    Adds simple injury-designation flags (Out/Doubtful/Questionable)
    for the home and away starting QB, primary RB, and top-2 WR/TE
    (by recent snap share), for each game. A missing injury-report
    entry is treated as healthy (flag = 0).

    injuries must include: season, week, team, gsis_id, position, report_status
    rb_stats / wrte_stats must already include offense_pct (snap share)
    """

    df_full = standardize_team_codes(df_full)

    qb_injuries = injuries[injuries['position'] == 'QB'].copy()
    qb_injuries['qb_injury_flag'] = qb_injuries['report_status'].isin(
        ['Out', 'Doubtful', 'Questionable']).astype(int)

    home_qb_inj = qb_injuries[['season', 'week', 'team', 'gsis_id', 'qb_injury_flag']].rename(
        columns={'team': 'home_team', 'gsis_id': 'home_qb_id', 'qb_injury_flag': 'home_qb_injury_flag'})
    away_qb_inj = qb_injuries[['season', 'week', 'team', 'gsis_id', 'qb_injury_flag']].rename(
        columns={'team': 'away_team', 'gsis_id': 'away_qb_id', 'qb_injury_flag': 'away_qb_injury_flag'})

    df_full = df_full.merge(home_qb_inj, on=['season', 'week', 'home_team', 'home_qb_id'], how='left')
    df_full = df_full.merge(away_qb_inj, on=['season', 'week', 'away_team', 'away_qb_id'], how='left')
    df_full['home_qb_injury_flag'] = df_full['home_qb_injury_flag'].fillna(0).astype(int)
    df_full['away_qb_injury_flag'] = df_full['away_qb_injury_flag'].fillna(0).astype(int)

    rb_sorted = rb_stats.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
    rb_sorted['recent_snap_pct'] = (
        rb_sorted.groupby(['player_id', 'season'])['offense_pct']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    primary_rb = (
        rb_sorted.sort_values('recent_snap_pct', ascending=False)
        .groupby(['team', 'game_id'], as_index=False)
        .first()[['team', 'game_id', 'season', 'week', 'player_id']]
    )

    rb_injuries = injuries[injuries['position'] == 'RB'].copy()
    rb_injuries['injury_flag'] = rb_injuries['report_status'].isin(
        ['Out', 'Doubtful', 'Questionable']).astype(int)
    rb_injuries = rb_injuries.rename(columns={'gsis_id': 'player_id'})

    primary_rb = primary_rb.merge(
        rb_injuries[['season', 'week', 'team', 'player_id', 'injury_flag']],
        on=['season', 'week', 'team', 'player_id'], how='left')
    primary_rb['injury_flag'] = primary_rb['injury_flag'].fillna(0).astype(int)
    primary_rb = primary_rb.rename(columns={'injury_flag': 'rb_injury_flag'})

    home_rb_inj = primary_rb[['season', 'week', 'team', 'rb_injury_flag']].rename(
        columns={'team': 'home_team_std', 'rb_injury_flag': 'home_rb_injury_flag'})
    away_rb_inj = primary_rb[['season', 'week', 'team', 'rb_injury_flag']].rename(
        columns={'team': 'away_team_std', 'rb_injury_flag': 'away_rb_injury_flag'})

    df_full = df_full.merge(home_rb_inj, on=['season', 'week', 'home_team_std'], how='left')
    df_full = df_full.merge(away_rb_inj, on=['season', 'week', 'away_team_std'], how='left')
    df_full['home_rb_injury_flag'] = df_full['home_rb_injury_flag'].fillna(0).astype(int)
    df_full['away_rb_injury_flag'] = df_full['away_rb_injury_flag'].fillna(0).astype(int)

    wrte_sorted = wrte_stats.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
    wrte_sorted['recent_snap_pct'] = (
        wrte_sorted.groupby(['player_id', 'season'])['offense_pct']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    top2_wrte = (
        wrte_sorted.sort_values('recent_snap_pct', ascending=False)
        .groupby(['team', 'game_id'])
        .head(2)[['team', 'game_id', 'season', 'week', 'player_id']]
    )

    wrte_injuries = injuries[injuries['position'].isin(['WR', 'TE'])].copy()
    wrte_injuries['injury_flag'] = wrte_injuries['report_status'].isin(
        ['Out', 'Doubtful', 'Questionable']).astype(int)
    wrte_injuries = wrte_injuries.rename(columns={'gsis_id': 'player_id'})

    top2_wrte = top2_wrte.merge(
        wrte_injuries[['season', 'week', 'team', 'player_id', 'injury_flag']],
        on=['season', 'week', 'team', 'player_id'], how='left')
    top2_wrte['injury_flag'] = top2_wrte['injury_flag'].fillna(0).astype(int)

    wrte_injury_flag = top2_wrte.groupby(
        ['team', 'game_id', 'season', 'week'], as_index=False)['injury_flag'].max()
    wrte_injury_flag = wrte_injury_flag.rename(columns={'injury_flag': 'wrte_injury_flag'})

    home_wrte_inj = wrte_injury_flag[['season', 'week', 'team', 'wrte_injury_flag']].rename(
        columns={'team': 'home_team_std', 'wrte_injury_flag': 'home_wrte_injury_flag'})
    away_wrte_inj = wrte_injury_flag[['season', 'week', 'team', 'wrte_injury_flag']].rename(
        columns={'team': 'away_team_std', 'wrte_injury_flag': 'away_wrte_injury_flag'})

    df_full = df_full.merge(home_wrte_inj, on=['season', 'week', 'home_team_std'], how='left')
    df_full = df_full.merge(away_wrte_inj, on=['season', 'week', 'away_team_std'], how='left')
    df_full['home_wrte_injury_flag'] = df_full['home_wrte_injury_flag'].fillna(0).astype(int)
    df_full['away_wrte_injury_flag'] = df_full['away_wrte_injury_flag'].fillna(0).astype(int)

    return df_full