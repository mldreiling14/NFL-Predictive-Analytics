from .common import standardize_team_codes


def add_elo_features(df_full, k_off_def=0.05, revert_fraction=1/3):
    """
    Adds split offense/defense rating features for both home and away
    teams, REPLACING the earlier single combined Elo rating. Each
    team gets two separate ratings: offense (points scored above/below
    league average, adjusted for opponent defense quality) and defense
    (points allowed below/above league average, adjusted for opponent
    offense quality). Both start at 0 (league average).

    This REPLACED margin-of-victory-adjusted combined Elo after
    confirming it performs BETTER when it fully replaces the combined
    rating (67.9% -> 68.2% in testing), but WORSE when added alongside
    the combined rating (67.9% -> 67.0%).

    Historically validated: 2016 Cleveland's defense (1-15 season)
    settles around -3.4 to -3.9, near the bottom of the real range
    (~-4.8 to +4.9 across the league).
    """
    df_full = standardize_team_codes(df_full)

    games = df_full[['game_id', 'season', 'gameday', 'home_team_std', 'away_team_std',
                      'home_score', 'away_score']].copy()
    games = games.sort_values(['season', 'gameday']).reset_index(drop=True)

    all_teams = set(games['home_team_std']).union(set(games['away_team_std']))
    league_avg_points = games[['home_score', 'away_score']].values.mean()

    off_rating = {team: 0.0 for team in all_teams}
    def_rating = {team: 0.0 for team in all_teams}
    current_season = None

    home_off_pre, home_def_pre, away_off_pre, away_def_pre = [], [], [], []

    for _, row in games.iterrows():
        if current_season is not None and row['season'] != current_season:
            for team in off_rating:
                off_rating[team] *= (1 - revert_fraction)
                def_rating[team] *= (1 - revert_fraction)
        current_season = row['season']

        h, a = row['home_team_std'], row['away_team_std']
        home_off_pre.append(off_rating[h])
        home_def_pre.append(def_rating[h])
        away_off_pre.append(off_rating[a])
        away_def_pre.append(def_rating[a])

        expected_home_score = league_avg_points + off_rating[h] - def_rating[a]
        expected_away_score = league_avg_points + off_rating[a] - def_rating[h]

        off_rating[h] += k_off_def * (row['home_score'] - expected_home_score)
        def_rating[a] -= k_off_def * (row['home_score'] - expected_home_score)

        off_rating[a] += k_off_def * (row['away_score'] - expected_away_score)
        def_rating[h] -= k_off_def * (row['away_score'] - expected_away_score)

    games['home_off_rating_pre'] = home_off_pre
    games['home_def_rating_pre'] = home_def_pre
    games['away_off_rating_pre'] = away_off_pre
    games['away_def_rating_pre'] = away_def_pre

    elo_merge = games[['game_id', 'home_off_rating_pre', 'home_def_rating_pre',
                        'away_off_rating_pre', 'away_def_rating_pre']]
    df_full = df_full.drop(columns=['home_elo_pre', 'away_elo_pre', 'home_off_rating_pre',
                                     'home_def_rating_pre', 'away_off_rating_pre', 'away_def_rating_pre'],
                            errors='ignore')
    df_full = df_full.merge(elo_merge, on='game_id', how='left')

    return df_full