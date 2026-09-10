def get_qb_game_log(player_id, qb_stats_raw, df_full, n=5):
    """Returns a QB's last N individual games with real per-game passing stats."""
    games = qb_stats_raw[qb_stats_raw['player_id'] == player_id].copy()
    if games.empty:
        import pandas as pd
        return pd.DataFrame()

    games = games.merge(
        df_full[['game_id', 'season', 'week', 'home_team', 'away_team', 'home_qb_id', 'away_qb_id']],
        on='game_id', how='left'
    )
    games['opponent'] = games.apply(
        lambda r: r['away_team'] if r['player_id'] == r['home_qb_id'] else r['home_team'], axis=1
    )

    games = games.sort_values('gameday', ascending=False).head(n)
    games = games.rename(columns={
        'passing_yards': 'Yards', 'passing_tds': 'TD', 'passing_interceptions': 'INT', 'passing_epa': 'EPA'
    })
    games['EPA'] = games['EPA'].round(2)

    return games[['season', 'week', 'opponent', 'Yards', 'TD', 'INT', 'EPA']].rename(
        columns={'season': 'Season', 'week': 'Week', 'opponent': 'Opp'}
    )