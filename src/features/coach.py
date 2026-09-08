import pandas as pd


def add_coach_features(df_full, window=5):
    """
    Adds two coach-related features for both home and away sides:
    1) Coach recent form (rolling win %, carries across seasons since
       it tracks the person, not the roster).
    2) Head-to-head record between the two coaches in this specific
       matchup, using only games played before this one. Games with
       no prior history get a neutral 0.5 (no known edge).
    """

    home_coach_games = df_full[['game_id', 'season', 'week', 'gameday', 'home_coach', 'home_win']].copy()
    home_coach_games = home_coach_games.rename(columns={'home_coach': 'coach'})
    home_coach_games['win'] = home_coach_games['home_win']

    away_coach_games = df_full[['game_id', 'season', 'week', 'gameday', 'away_coach', 'home_win']].copy()
    away_coach_games = away_coach_games.rename(columns={'away_coach': 'coach'})
    away_coach_games['win'] = 1 - away_coach_games['home_win']

    coach_games = pd.concat([home_coach_games, away_coach_games], ignore_index=True)
    coach_games = coach_games.sort_values(['coach', 'gameday']).reset_index(drop=True)

    coach_games['recent_coach_form'] = (
        coach_games.groupby('coach')['win']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )

    home_rows = coach_games.merge(df_full[['game_id', 'home_coach']],
                                   left_on=['game_id', 'coach'], right_on=['game_id', 'home_coach'])
    away_rows = coach_games.merge(df_full[['game_id', 'away_coach']],
                                   left_on=['game_id', 'coach'], right_on=['game_id', 'away_coach'])

    home_coach_recent = home_rows[['game_id', 'recent_coach_form']].rename(
        columns={'recent_coach_form': 'home_coach_recent_form'})
    away_coach_recent = away_rows[['game_id', 'recent_coach_form']].rename(
        columns={'recent_coach_form': 'away_coach_recent_form'})

    df_full = df_full.drop(columns=['home_coach_recent_form', 'away_coach_recent_form'], errors='ignore')
    df_full = df_full.merge(home_coach_recent, on='game_id', how='left')
    df_full = df_full.merge(away_coach_recent, on='game_id', how='left')

    coach_matchups = df_full[['game_id', 'season', 'week', 'gameday', 'home_coach', 'away_coach', 'home_win']].copy()
    coach_matchups['matchup_key'] = coach_matchups.apply(
        lambda r: tuple(sorted([r['home_coach'], r['away_coach']])), axis=1
    )
    coach_matchups = coach_matchups.sort_values('gameday').reset_index(drop=True)

    results = []
    for key, group in coach_matchups.groupby('matchup_key'):
        group = group.sort_values('gameday').reset_index(drop=True)
        for i in range(len(group)):
            row = group.iloc[i]
            prior = group.iloc[:i]
            if len(prior) == 0:
                results.append({'game_id': row['game_id'], 'home_coach_h2h_wins': None, 'h2h_games_played': 0})
                continue

            home_coach_wins = (
                ((prior['home_coach'] == row['home_coach']) & (prior['home_win'] == 1)) |
                ((prior['away_coach'] == row['home_coach']) & (prior['home_win'] == 0))
            ).sum()

            results.append({
                'game_id': row['game_id'],
                'home_coach_h2h_wins': home_coach_wins / len(prior),
                'h2h_games_played': len(prior)
            })

    h2h_results = pd.DataFrame(results)

    df_full = df_full.drop(columns=['home_coach_h2h_wins', 'h2h_games_played'], errors='ignore')
    df_full = df_full.merge(h2h_results, on='game_id', how='left')
    df_full['home_coach_h2h_wins'] = df_full['home_coach_h2h_wins'].fillna(0.5)
    df_full['h2h_games_played'] = df_full['h2h_games_played'].fillna(0)

    return df_full