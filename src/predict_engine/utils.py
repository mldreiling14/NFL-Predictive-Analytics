def moneyline_to_prob(ml):
    """Converts American moneyline odds to implied win probability. Not de-vigged -
    the two sides' implied probabilities will sum to slightly over 100%, reflecting
    the sportsbook's built-in margin."""
    import pandas as pd
    if pd.isna(ml):
        return None
    if ml < 0:
        return -ml / (-ml + 100)
    else:
        return 100 / (ml + 100)