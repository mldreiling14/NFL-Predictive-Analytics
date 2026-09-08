# Franchises that relocated during this dataset's date range —
# schedule data uses the old code, but player-stats data uses the new one.
TEAM_CODE_MAP = {
    'SD': 'LAC',   # San Diego -> LA Chargers
    'STL': 'LA',   # St. Louis -> LA Rams
    'OAK': 'LV'    # Oakland -> Las Vegas
}


def standardize_team_codes(df_full):
    """
    Adds home_team_std / away_team_std columns that translate old,
    relocated-franchise team codes into the codes used by player-level
    stats tables, without altering the original home_team/away_team
    columns used for display.
    """
    df_full = df_full.copy()
    df_full['home_team_std'] = df_full['home_team'].replace(TEAM_CODE_MAP)
    df_full['away_team_std'] = df_full['away_team'].replace(TEAM_CODE_MAP)
    return df_full