import pandas as pd


def add_recent_form_features(df_full, window=5):
    """
    Adds rolling recent-form features (win %, point differential)
    for both home and away teams, using only games prior to each
    game (no data leakage), reset at each season boundary.
    """

    home = df_full[['game_id', 'season', 'week', 'gameday',
                     'home_team', 'away_team', 'home_win',
                     'home_score', 'away_score']].copy()
    home = home.rename(columns={'home_team': 'team', 'away_team': 'opponent'})
    home['win'] = home['home_win']
    home['is_home'] = 1
    home['points_for'] = home['home_score']
    home['points_against'] = home['away_score']

    away = df_full[['game_id', 'season', 'week', 'gameday',
                     'home_team', 'away_team', 'home_win',
                     'home_score', 'away_score']].copy()
    away = away.rename(columns={'away_team': 'team', 'home_team': 'opponent'})
    away['win'] = 1 - away['home_win']
    away['is_home'] = 0
    away['points_for'] = away['away_score']
    away['points_against'] = away['home_score']

    team_games = pd.concat([home, away], ignore_index=True)
    team_games = team_games.sort_values(['team', 'season', 'gameday']).reset_index(drop=True)
    team_games['point_diff'] = team_games['points_for'] - team_games['points_against']

    team_games['recent_form'] = (
        team_games.groupby(['team', 'season'])['win']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )

    team_games['recent_point_diff'] = (
        team_games.groupby(['team', 'season'])['point_diff']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )

    home_form = team_games[team_games['is_home'] == 1][['game_id', 'recent_form']].rename(
        columns={'recent_form': 'home_recent_form'})
    away_form = team_games[team_games['is_home'] == 0][['game_id', 'recent_form']].rename(
        columns={'recent_form': 'away_recent_form'})

    home_pd = team_games[team_games['is_home'] == 1][['game_id', 'recent_point_diff']].rename(
        columns={'recent_point_diff': 'home_recent_point_diff'})
    away_pd = team_games[team_games['is_home'] == 0][['game_id', 'recent_point_diff']].rename(
        columns={'recent_point_diff': 'away_recent_point_diff'})

    df_full = df_full.drop(columns=[
        'home_recent_form', 'away_recent_form',
        'home_recent_point_diff', 'away_recent_point_diff'
    ], errors='ignore')

    df_full = df_full.merge(home_form, on='game_id', how='left')
    df_full = df_full.merge(away_form, on='game_id', how='left')
    df_full = df_full.merge(home_pd, on='game_id', how='left')
    df_full = df_full.merge(away_pd, on='game_id', how='left')

    return df_full