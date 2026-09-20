# scripts/explain_prediction.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import pandas as pd
import nflreadpy as nfl
from predict_engine import load_model, build_snapshots
from predict_engine.matchup import build_matchup_features

model_bundle = load_model(model_path='models/win_probability_model.joblib')
snapshots = build_snapshots(db_path='data/nfl.db', seasons=range(2024, 2027))

sched = nfl.load_schedules(seasons=[2026]).to_pandas()
game = sched[sched['game_id'] == '2026_02_MIA_SF'].iloc[0]

feature_row = build_matchup_features(
    game['home_team'], game['away_team'], snapshots,
    game['home_rest'], game['away_rest'], game['div_game'], game['spread_line']
)
feature_cols = model_bundle['feature_cols']
X = feature_row[feature_cols]
X_imputed = pd.DataFrame(model_bundle['imputer'].transform(X), columns=feature_cols)

model = model_bundle['model']
print(type(model))
if hasattr(model, 'named_steps'):
    print(model.named_steps)
prob = model.predict_proba(X_imputed)[0][1]
print(f"Predicted home ({game['home_team']}) win probability: {prob:.1%}\n")

scaler = model.named_steps['standardscaler']
logreg = model.named_steps['logisticregression']

coefs = logreg.coef_[0]
intercept = logreg.intercept_[0]
X_scaled = scaler.transform(X_imputed)

contributions = coefs * X_scaled[0]
results = pd.DataFrame({
    'feature': feature_cols,
    'raw_value': X_imputed.iloc[0].values,
    'coefficient': coefs,
    'contribution_to_logit': contributions,
}).sort_values('contribution_to_logit', key=abs, ascending=False)

print(f"Intercept (baseline logit, home-field edge baked in): {intercept:.3f}\n")
print("Top 15 features by contribution to the home team's (ATL) predicted logit:")
print(results.head(15).to_string(index=False))