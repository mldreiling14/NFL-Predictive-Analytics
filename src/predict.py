from predict_engine import load_model, build_snapshots, get_injury_report, get_qb_game_log, \
    build_matchup_features, predict_matchup, predict_week, moneyline_to_prob, DETAIL_COLS


if __name__ == "__main__":
    model_bundle = load_model()
    snapshots = build_snapshots()
    predictions = predict_week(2026, 1, snapshots, model_bundle)
    print(predictions)