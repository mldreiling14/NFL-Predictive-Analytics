from .common import standardize_team_codes


def add_wrte_features(df_full, wrte_stats, window=5):
    """
    Adds rolling team-level WR+TE receiving-corps performance features
    (snap-share-weighted receiving yards, EPA, targets) for both home
    and away teams. Aggregates all WRs and TEs on a team into one
    team-level signal per game, then computes a rolling average over
    that team's prior games (no leakage), reset at each season boundary.

    wrte_stats must already include: team, game_id, season, week,
    targets, receiving_yards, receiving_epa, offense_pct
    """

    wrte_stats = wrte_stats.copy()
    wrte_stats['weighted_rec_yards'] = wrte_stats['receiving_yards'] * wrte_stats['offense_pct']
    wrte_stats['weighted_rec_epa'] = wrte_stats['receiving_epa'] * wrte_stats['offense_pct']
    wrte_stats['weighted_targets'] = wrte_stats['targets'] * wrte_stats['offense_pct']

    wrte_team_game = wrte_stats.groupby(['team', 'game_id', 'season', 'week'], as_index=False).agg(
        team_weighted_rec_yards=('weighted_rec_yards', 'sum'),
        team_weighted_rec_epa=('weighted_rec_epa', 'sum'),
        team_weighted_targets=('weighted_targets', 'sum')
    )

    wrte_team_game = wrte_team_game.sort_values(['team', 'season', 'week']).reset_index(drop=True)
    for col in ['team_weighted_rec_yards', 'team_weighted_rec_epa', 'team_weighted_targets']:
        wrte_team_game[f'recent_{col}'] = (
            wrte_team_game.groupby(['team', 'season'])[col]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )

    df_full = standardize_team_codes(df_full)

    wrte_recent_cols = ['team', 'game_id', 'recent_team_weighted_rec_yards',
                         'recent_team_weighted_rec_epa', 'recent_team_weighted_targets']
    wrte_recent = wrte_team_game[wrte_recent_cols]

    home_wrte = wrte_recent.rename(columns={
        'team': 'home_team_std',
        'recent_team_weighted_rec_yards': 'home_wrte_recent_rec_yards',
        'recent_team_weighted_rec_epa': 'home_wrte_recent_rec_epa',
        'recent_team_weighted_targets': 'home_wrte_recent_targets'
    })

    away_wrte = wrte_recent.rename(columns={
        'team': 'away_team_std',
        'recent_team_weighted_rec_yards': 'away_wrte_recent_rec_yards',
        'recent_team_weighted_rec_epa': 'away_wrte_recent_rec_epa',
        'recent_team_weighted_targets': 'away_wrte_recent_targets'
    })

    df_full = df_full.drop(columns=[c for c in df_full.columns if 'wrte_recent' in c], errors='ignore')

    df_full = df_full.merge(home_wrte, on=['home_team_std', 'game_id'], how='left')
    df_full = df_full.merge(away_wrte, on=['away_team_std', 'game_id'], how='left')

    return df_full