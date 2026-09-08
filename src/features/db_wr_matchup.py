def add_db_wr_matchup_features(df_full, player_stats, snaps_full, crosswalk, physical, adv_def_full, window=5):
    """
    Adds an approximated WR-vs-CB size and coverage matchup feature.
    NFL data does not publish which specific CB covers which specific
    WR - this is "team's best WR" vs "opponent's most-used CB".

    BUG HISTORY: the CB identification step MUST use an INNER merge
    (not left) between adv_def_full and cb_snaps, or non-CB defenders
    can win the "primary CB" selection on ties (e.g. Week 1). Produced
    a real bug (a 300lb "cornerback" who was actually a defensive end).

    Only available for 2018+ (PFR advanced stats limitation).

    adv_def_full's team column follows PFR's actual-code-at-the-time
    convention - merge against df_full's raw home_team/away_team.
    """

    cb_snaps = snaps_full[snaps_full['position'] == 'CB'][['pfr_player_id', 'game_id', 'team', 'defense_pct']]

    cb_stats = adv_def_full.merge(
        cb_snaps[['pfr_player_id', 'game_id', 'defense_pct']],
        on=['pfr_player_id', 'game_id'], how='inner'
    )

    cb_stats = cb_stats.sort_values(['pfr_player_id', 'season', 'week']).reset_index(drop=True)
    cb_stats['recent_defense_pct'] = (
        cb_stats.groupby(['pfr_player_id', 'season'])['defense_pct']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    for col in ['def_completion_pct', 'def_passer_rating_allowed']:
        cb_stats[f'recent_{col}'] = (
            cb_stats.groupby(['pfr_player_id', 'season'])[col]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )

    primary_cb = (
        cb_stats.sort_values('recent_defense_pct', ascending=False)
        .groupby(['team', 'game_id'], as_index=False)
        .first()[['team', 'game_id', 'season', 'week', 'pfr_player_id',
                  'recent_def_completion_pct', 'recent_def_passer_rating_allowed']]
    )
    primary_cb = primary_cb.merge(physical, on='pfr_player_id', how='left')

    wr_snaps = snaps_full[snaps_full['position'] == 'WR'][['pfr_player_id', 'game_id', 'team', 'offense_pct']]

    wr_targets = player_stats[player_stats['position'] == 'WR'][
        ['player_id', 'game_id', 'season', 'week', 'team', 'targets']].copy()
    wr_targets = wr_targets.merge(crosswalk, on='player_id', how='left')
    wr_targets = wr_targets.merge(
        wr_snaps[['pfr_player_id', 'game_id', 'offense_pct']], on=['pfr_player_id', 'game_id'], how='left')

    wr_targets = wr_targets.sort_values(['pfr_player_id', 'season', 'week']).reset_index(drop=True)
    wr_targets['recent_offense_pct'] = (
        wr_targets.groupby(['pfr_player_id', 'season'])['offense_pct']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )

    primary_wr = (
        wr_targets.sort_values('recent_offense_pct', ascending=False)
        .groupby(['team', 'game_id'], as_index=False)
        .first()[['team', 'game_id', 'season', 'week', 'pfr_player_id']]
    )
    primary_wr = primary_wr.merge(physical, on='pfr_player_id', how='left')

    wr_for_matchup = primary_wr[['team', 'game_id', 'season', 'height', 'weight']].rename(
        columns={'height': 'wr_height', 'weight': 'wr_weight'})

    cb_for_matchup = primary_cb[['team', 'game_id', 'height', 'weight',
                                   'recent_def_completion_pct', 'recent_def_passer_rating_allowed']].rename(
        columns={'team': 'opponent_team', 'height': 'cb_height', 'weight': 'cb_weight'})

    game_opponents = df_full[['game_id', 'home_team', 'away_team']].copy()

    matchup = wr_for_matchup.merge(game_opponents, on='game_id', how='left')
    matchup['opponent_team'] = matchup.apply(
        lambda r: r['away_team'] if r['team'] == r['home_team'] else r['home_team'], axis=1)

    matchup = matchup.merge(cb_for_matchup, on=['game_id', 'opponent_team'], how='left')
    matchup['height_advantage'] = matchup['wr_height'] - matchup['cb_height']
    matchup['weight_advantage'] = matchup['wr_weight'] - matchup['cb_weight']

    matchup_final = matchup[['team', 'game_id', 'height_advantage', 'weight_advantage',
                              'recent_def_completion_pct', 'recent_def_passer_rating_allowed']].rename(
        columns={'recent_def_completion_pct': 'opp_cb_completion_pct_allowed',
                 'recent_def_passer_rating_allowed': 'opp_cb_rating_allowed'})

    home_matchup = matchup_final.rename(columns={
        'team': 'home_team',
        'height_advantage': 'home_wr_height_advantage',
        'weight_advantage': 'home_wr_weight_advantage',
        'opp_cb_completion_pct_allowed': 'home_opp_cb_completion_allowed',
        'opp_cb_rating_allowed': 'home_opp_cb_rating_allowed'
    })
    away_matchup = matchup_final.rename(columns={
        'team': 'away_team',
        'height_advantage': 'away_wr_height_advantage',
        'weight_advantage': 'away_wr_weight_advantage',
        'opp_cb_completion_pct_allowed': 'away_opp_cb_completion_allowed',
        'opp_cb_rating_allowed': 'away_opp_cb_rating_allowed'
    })

    df_full = df_full.drop(columns=[c for c in df_full.columns if
                                     'wr_height_advantage' in c or 'wr_weight_advantage' in c or 'opp_cb' in c],
                            errors='ignore')
    df_full = df_full.merge(home_matchup, on=['game_id', 'home_team'], how='left')
    df_full = df_full.merge(away_matchup, on=['game_id', 'away_team'], how='left')

    return df_full