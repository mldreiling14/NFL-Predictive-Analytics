def add_qb_features(df_full, qb_stats, window=5):
    """
    Adds rolling QB performance features (yards, TDs, INTs, EPA)
    for both home and away starting QBs, using only that QB's
    prior starts (no data leakage). Does NOT reset at season
    boundaries, since a QB's own performance carries across
    the offseason (unlike team-level roster turnover).
    """

    qb_stats = qb_stats.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

    for col in ['passing_yards', 'passing_tds', 'passing_interceptions', 'passing_epa']:
        qb_stats[f'recent_{col}'] = (
            qb_stats.groupby('player_id')[col]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )

    qb_recent_cols = ['player_id', 'game_id', 'recent_passing_yards', 'recent_passing_tds',
                       'recent_passing_interceptions', 'recent_passing_epa']
    qb_recent = qb_stats[qb_recent_cols]

    home_qb = qb_recent.rename(columns={
        'player_id': 'home_qb_id',
        'recent_passing_yards': 'home_qb_recent_yards',
        'recent_passing_tds': 'home_qb_recent_tds',
        'recent_passing_interceptions': 'home_qb_recent_ints',
        'recent_passing_epa': 'home_qb_recent_epa'
    })

    away_qb = qb_recent.rename(columns={
        'player_id': 'away_qb_id',
        'recent_passing_yards': 'away_qb_recent_yards',
        'recent_passing_tds': 'away_qb_recent_tds',
        'recent_passing_interceptions': 'away_qb_recent_ints',
        'recent_passing_epa': 'away_qb_recent_epa'
    })

    df_full = df_full.drop(columns=[c for c in df_full.columns if 'qb_recent' in c], errors='ignore')

    df_full = df_full.merge(home_qb, on=['home_qb_id', 'game_id'], how='left')
    df_full = df_full.merge(away_qb, on=['away_qb_id', 'game_id'], how='left')

    return df_full