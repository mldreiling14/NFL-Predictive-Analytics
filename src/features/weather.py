def add_weather_features(df_full):
    """
    Adds weather features: dome-neutral temp/wind, cold/high-wind
    flags, and "climate shock" (visiting team facing much colder
    conditions than their home norm). Real signal confirmed
    (61.6% vs 54.8% home win rate in cold-shock games) but not
    included in the production model - see FEATURES.md.

    Requires temp, wind, roof, home_team_std, away_team_std, home_win
    already present in df_full.
    """
    df_full = df_full.copy()

    df_full['is_outdoor'] = df_full['roof'].isin(['outdoors', 'open']).astype(int)
    df_full['temp_adj'] = df_full['temp'].where(df_full['is_outdoor'] == 1, 70)
    df_full['wind_adj'] = df_full['wind'].where(df_full['is_outdoor'] == 1, 0)

    df_full['cold_game'] = ((df_full['is_outdoor'] == 1) & (df_full['temp_adj'] <= 32)).astype(int)
    df_full['high_wind_game'] = ((df_full['is_outdoor'] == 1) & (df_full['wind_adj'] >= 15)).astype(int)

    team_climate = df_full[df_full['is_outdoor'] == 1].groupby('home_team_std')['temp_adj'].mean()
    df_full['away_home_climate'] = df_full['away_team_std'].map(team_climate)
    df_full['away_home_climate'] = df_full['away_home_climate'].fillna(70)

    df_full['climate_shock'] = None
    outdoor_mask = df_full['is_outdoor'] == 1
    df_full.loc[outdoor_mask, 'climate_shock'] = (
        df_full.loc[outdoor_mask, 'away_home_climate'] - df_full.loc[outdoor_mask, 'temp_adj']
    )
    df_full['climate_shock'] = df_full['climate_shock'].fillna(0).astype(float)

    df_full['cold_shock_game'] = ((df_full['climate_shock'] >= 25) & (df_full['is_outdoor'] == 1)).astype(int)

    return df_full