from .common import standardize_team_codes


def add_rb_features(df_full, rb_stats, window=5):
    """
    Adds rolling team-level RB-group performance features (snap-share-
    weighted rushing yards, rushing EPA, receiving yards) for both home
    and away teams. Aggregates all RBs on a team into one team-level
    signal per game, then computes a rolling average over that team's
    prior games (no leakage), reset at each season boundary.

    rb_stats must already include: team, game_id, season, week,
    carries, rushing_yards, rushing_epa, receiving_yards, offense_pct
    """

    rb_stats = rb_stats.copy()
    rb_stats['weighted_rush_yards'] = rb_stats['rushing_yards'] * rb_stats['offense_pct']
    rb_stats['weighted_rush_epa'] = rb_stats['rushing_epa'] * rb_stats['offense_pct']
    rb_stats['weighted_rec_yards'] = rb_stats['receiving_yards'] * rb_stats['offense_pct']

    rb_team_game = rb_stats.groupby(['team', 'game_id', 'season', 'week'], as_index=False).agg(
        team_weighted_rush_yards=('weighted_rush_yards', 'sum'),
        team_weighted_rush_epa=('weighted_rush_epa', 'sum'),
        team_weighted_rec_yards=('weighted_rec_yards', 'sum'),
        team_rb_carries=('carries', 'sum')
    )

    rb_team_game = rb_team_game.sort_values(['team', 'season', 'week']).reset_index(drop=True)
    for col in ['team_weighted_rush_yards', 'team_weighted_rush_epa', 'team_weighted_rec_yards']:
        rb_team_game[f'recent_{col}'] = (
            rb_team_game.groupby(['team', 'season'])[col]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )

    df_full = standardize_team_codes(df_full)

    rb_recent_cols = ['team', 'game_id', 'recent_team_weighted_rush_yards',
                       'recent_team_weighted_rush_epa', 'recent_team_weighted_rec_yards']
    rb_recent = rb_team_game[rb_recent_cols]

    home_rb = rb_recent.rename(columns={
        'team': 'home_team_std',
        'recent_team_weighted_rush_yards': 'home_rb_recent_rush_yards',
        'recent_team_weighted_rush_epa': 'home_rb_recent_rush_epa',
        'recent_team_weighted_rec_yards': 'home_rb_recent_rec_yards'
    })

    away_rb = rb_recent.rename(columns={
        'team': 'away_team_std',
        'recent_team_weighted_rush_yards': 'away_rb_recent_rush_yards',
        'recent_team_weighted_rush_epa': 'away_rb_recent_rush_epa',
        'recent_team_weighted_rec_yards': 'away_rb_recent_rec_yards'
    })

    df_full = df_full.drop(columns=[c for c in df_full.columns if 'rb_recent' in c], errors='ignore')

    df_full = df_full.merge(home_rb, on=['home_team_std', 'game_id'], how='left')
    df_full = df_full.merge(away_rb, on=['away_team_std', 'game_id'], how='left')

    return df_full