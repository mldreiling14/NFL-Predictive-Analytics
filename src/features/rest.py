def add_rest_advantage(df_full):
    """
    Adds rest_advantage: the difference in days of rest between the
    home and away team (positive = home team had more rest).
    """
    df_full = df_full.copy()
    df_full['rest_advantage'] = df_full['home_rest'] - df_full['away_rest']
    return df_full