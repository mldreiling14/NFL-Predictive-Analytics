from .common import standardize_team_codes


def add_oline_features(df_full, oline_stats, pressure_stats):
    """
    Adds O-line/pass-protection features: sack rate (full 2015+ range,
    standardized codes) and QB pressure rate (2018+, RAW team codes -
    PFR uses the actual code at the time, opposite of load_team_stats'
    convention of backfilling to current codes).

    oline_stats must include: game_id, season, week, team, attempts, sacks_suffered
    pressure_stats must include: game_id, season, week, team, times_pressured_pct
    """

    oline_stats = oline_stats.copy()
    oline_stats['sack_rate'] = oline_stats['sacks_suffered'] / (oline_stats['attempts'] + oline_stats['sacks_suffered'])
    oline_stats = oline_stats.sort_values(['team', 'season', 'week']).reset_index(drop=True)
    oline_stats['recent_sack_rate'] = (
        oline_stats.groupby(['team', 'season'])['sack_rate']
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    )

    df_full = standardize_team_codes(df_full)

    sack_rate_recent = oline_stats[['team', 'game_id', 'recent_sack_rate']]
    home_sr = sack_rate_recent.rename(columns={'team': 'home_team_std', 'recent_sack_rate': 'home_sack_rate_recent'})
    away_sr = sack_rate_recent.rename(columns={'team': 'away_team_std', 'recent_sack_rate': 'away_sack_rate_recent'})

    df_full = df_full.drop(columns=['home_sack_rate_recent', 'away_sack_rate_recent'], errors='ignore')
    df_full = df_full.merge(home_sr, on=['home_team_std', 'game_id'], how='left')
    df_full = df_full.merge(away_sr, on=['away_team_std', 'game_id'], how='left')

    pressure_stats = pressure_stats.copy()
    pressure_team_game = pressure_stats.groupby(['team', 'game_id', 'season', 'week'], as_index=False).mean(numeric_only=True)
    pressure_team_game = pressure_team_game.sort_values(['team', 'season', 'week']).reset_index(drop=True)
    pressure_team_game['recent_pressure_pct'] = (
        pressure_team_game.groupby(['team', 'season'])['times_pressured_pct']
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    )

    pressure_recent = pressure_team_game[['team', 'game_id', 'recent_pressure_pct']]
    home_press = pressure_recent.rename(columns={'team': 'home_team', 'recent_pressure_pct': 'home_pressure_pct_recent'})
    away_press = pressure_recent.rename(columns={'team': 'away_team', 'recent_pressure_pct': 'away_pressure_pct_recent'})

    df_full = df_full.drop(columns=['home_pressure_pct_recent', 'away_pressure_pct_recent'], errors='ignore')
    df_full = df_full.merge(home_press, on=['home_team', 'game_id'], how='left')
    df_full = df_full.merge(away_press, on=['away_team', 'game_id'], how='left')

    return df_full