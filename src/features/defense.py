from .common import standardize_team_codes


def add_defense_allowed_features(df_full, team_stats, window=5):
    """
    Adds rolling team-level defensive features based on what a team's
    defense ALLOWED (opponent's own offensive EPA/yards that game) and
    takeaways forced (INTs + fumble recoveries combined).

    team_stats must include: game_id, season, week, team, opponent_team,
    passing_epa, rushing_epa, passing_yards, rushing_yards,
    def_interceptions, fumble_recovery_opp

    Note: team_stats already uses current franchise codes (LAC/LA/LV),
    so it merges directly against df_full's standardized team columns.
    """

    team_stats = team_stats[['game_id', 'season', 'week', 'team', 'opponent_team',
                              'passing_epa', 'rushing_epa', 'passing_yards', 'rushing_yards',
                              'def_interceptions', 'fumble_recovery_opp']].copy()

    opponent_offense = team_stats[['game_id', 'team', 'passing_epa', 'rushing_epa',
                                    'passing_yards', 'rushing_yards']].rename(
        columns={'team': 'opponent_team', 'passing_epa': 'opp_passing_epa',
                 'rushing_epa': 'opp_rushing_epa', 'passing_yards': 'opp_passing_yards',
                 'rushing_yards': 'opp_rushing_yards'}
    )

    team_stats = team_stats.merge(opponent_offense, on=['game_id', 'opponent_team'], how='left')

    team_stats['epa_allowed'] = team_stats['opp_passing_epa'] + team_stats['opp_rushing_epa']
    team_stats['yards_allowed'] = team_stats['opp_passing_yards'] + team_stats['opp_rushing_yards']
    team_stats['takeaways'] = team_stats['def_interceptions'] + team_stats['fumble_recovery_opp']

    team_stats = team_stats.sort_values(['team', 'season', 'week']).reset_index(drop=True)
    for col in ['epa_allowed', 'yards_allowed', 'takeaways']:
        team_stats[f'recent_{col}'] = (
            team_stats.groupby(['team', 'season'])[col]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )

    df_full = standardize_team_codes(df_full)

    def_allowed_recent = team_stats[['team', 'game_id', 'recent_epa_allowed',
                                       'recent_yards_allowed', 'recent_takeaways']]

    home_def = def_allowed_recent.rename(columns={
        'team': 'home_team_std',
        'recent_epa_allowed': 'home_epa_allowed_recent',
        'recent_yards_allowed': 'home_yards_allowed_recent',
        'recent_takeaways': 'home_takeaways_recent'
    })
    away_def = def_allowed_recent.rename(columns={
        'team': 'away_team_std',
        'recent_epa_allowed': 'away_epa_allowed_recent',
        'recent_yards_allowed': 'away_yards_allowed_recent',
        'recent_takeaways': 'away_takeaways_recent'
    })

    df_full = df_full.drop(columns=[c for c in df_full.columns if
                                     'epa_allowed_recent' in c or 'yards_allowed_recent' in c or 'takeaways_recent' in c],
                            errors='ignore')
    df_full = df_full.merge(home_def, on=['home_team_std', 'game_id'], how='left')
    df_full = df_full.merge(away_def, on=['away_team_std', 'game_id'], how='left')

    return df_full