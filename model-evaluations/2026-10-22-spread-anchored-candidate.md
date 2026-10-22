# Candidate (spread-anchored + prior-blended features) vs production, walk-forward 2024-2025

## Context
- Production: main working tree, `games_with_features` (unblended team stats), SimpleImputer(mean) + StandardScaler + LogisticRegression(max_iter=1000), 61 features.
- Candidate: worktree `sports_betting-candidate` (branch candidate/spread-anchored). Team rolling stats blended with last season (K=4), plus SpreadAnchoredModel (C=0.0003, spread x10). Fitted with its `train_model.fit_model`, using that worktree's own db.
- References: LogisticRegression on spread_line alone (default C=1, no scaling), and the Vegas favorite (spread_line > 0 = home favored; no pushes or missing spreads in the test sets).
- C was tuned on 2021-2023 only, so 2024+ is untouched holdout. No earlier reports existed in `model-evaluations/` (folder created), so this format is new.
- Method: for each (season, week) test week, each recipe is refit on completed games strictly before that week, from its own db, then predicts that week. Saved .joblib artifacts were not scored. Both dbs hold the same 3121 completed games (game_id sets verified equal; spread_line identical; the spread-only reference is identical across dbs).
- Test set: 2024 (285 games, 22 weeks incl. playoffs), 2025 (285, 22 weeks), so 570 games / 44 weeks. Bonus: 2026 weeks 1-6 (93 games). Ties would count as home losses (pipeline label rule).
- Caveats: the main working tree has uncommitted changes (src/features/*, data/nfl.db), so "production" is the current working-tree features, not HEAD. Home win rate in 2024-25 was 54.0%; a constant base-rate predictor has log loss 0.6899.

## Results
2024-2025 combined (n=570)

| Model | Accuracy | Log loss | Brier |
|---|---|---|---|
| Production | 0.6719 | 0.6022 | 0.2075 |
| Candidate | 0.6825 | 0.5999 | 0.2067 |
| Spread-only LR | 0.6825 | 0.5978 | 0.2060 |
| Vegas favorite | 0.6825 | n/a | n/a |

Per season

| Season | Model | Acc | LL | Brier |
|---|---|---|---|---|
| 2024 | Production | 0.6947 | 0.5904 | 0.2011 |
| 2024 | Candidate | 0.7053 | 0.5902 | 0.2014 |
| 2024 | Spread-only | 0.7053 | 0.5890 | 0.2011 |
| 2025 | Production | 0.6491 | 0.6140 | 0.2139 |
| 2025 | Candidate | 0.6596 | 0.6097 | 0.2120 |
| 2025 | Spread-only | 0.6596 | 0.6066 | 0.2109 |
| 2026 wk1-6 (bonus, n=93) | Production | 0.6237 | 0.6865 | 0.2433 |
| 2026 wk1-6 | Candidate | 0.6237 | 0.6635 | 0.2345 |
| 2026 wk1-6 | Spread-only | 0.6237 | 0.6636 | 0.2342 |

Paired per-game differences (candidate minus comparator; negative = candidate better), bootstrap 95% CI over games, 5000 resamples

| Sample | dLogLoss vs prod | dBrier vs prod | dLogLoss vs spread-only | dBrier vs spread-only |
|---|---|---|---|---|
| 2024-25 | -0.00225 [-0.0155, +0.0108] | -0.00081 [-0.0065, +0.0046] | +0.00213 [-0.0003, +0.0044] | +0.00070 [-0.0003, +0.0018] |
| 2024 | -0.00023 [-0.0180, +0.0170] | +0.00029 [-0.0072, +0.0078] | +0.00115 [-0.0022, +0.0044] | +0.00026 [-0.0012, +0.0017] |
| 2025 | -0.00428 [-0.0233, +0.0144] | -0.00190 [-0.0103, +0.0063] | +0.00312 [-0.0003, +0.0065] | +0.00115 [-0.0003, +0.0026] |
| 2026 wk1-6 | -0.02302 [-0.0665, +0.0157] | -0.00881 [-0.0270, +0.0078] | -0.00007 [-0.0059, +0.0054] | +0.00026 [-0.0020, +0.0025] |

Accuracy: candidate = Vegas favorite = spread-only in every sample (identical picks). The candidate beats production by 6 games of 570 (+1.05 pts), well within noise.

Weeks won (lower weekly log loss; 2024-25 = 44 weeks)
- Candidate vs production: 21 of 44 by log loss (2024: 8/22, 2025: 13/22), 21 of 44 by Brier.
- Candidate vs spread-only: 17 of 44 (2024: 10/22, 2025: 7/22).
- 2026 wk1-6: candidate beats production 5/6 and spread-only 5/6.

Robustness to extreme weeks (2024-25)
- Dropping the best week (2024 wk2, dLL -0.133 per game) and the worst week (2024 wk22, a single game): dLL vs production becomes +0.00135 [-0.0117, +0.0141], so the sign flips. For Brier (dropping best/worst Brier weeks): -0.00067 [-0.0063, +0.0049].
- Versus spread-only, dropping the extremes leaves +0.00215, so that deficit is not driven by outliers.

Weeks 1-4 vs 5+ (2024-25)

| Split | n | Prod LL | Cand LL | Spread LL | dLL cand-prod | dLL cand-spread |
|---|---|---|---|---|---|---|
| Wk 1-4 | 128 | 0.6436 | 0.6389 | 0.6394 | -0.00471 [-0.034, +0.025] | -0.00048 [-0.005, +0.004] |
| Wk 5+ | 442 | 0.5902 | 0.5887 | 0.5858 | -0.00154 [-0.017, +0.013] | +0.00289 [+0.0002, +0.0055] |
| Wk 5-18 only | 416 | 0.5895 | 0.5875 | 0.5845 | -0.00200 | +0.00298 [+0.0001, +0.0058] |
| Playoffs (wk 19+) | 26 | 0.6022 | 0.6080 | 0.6064 | +0.00573 | +0.00151 |

Brier gives the same ordering. In weeks 1-4 the candidate won only 3 of 8 early weeks against production (2024: 2/4, 2025: 1/4); in 2026 wk1-4 it won 3/4. 2026 wk1-4 dLL -0.027 [-0.086, +0.028]; wk5-6 -0.014.

## Week-by-week notes
- Weekly log-loss differences swing between roughly -0.13 and +0.10; weeks have 14-16 games (playoff weeks 1-6), so single weeks mean little. Weekly numbers are in the scratchpad `weekly.csv` (outside the repo).
- 2024 wk2 is the biggest candidate win (production 0.936 vs candidate 0.803 LL; the Vegas favorite missed 9 of 16). Big candidate wins tend to be high-upset weeks, where production's departures from the spread get punished. 2025 wk3 is the biggest candidate loss (+0.081).
- High Vegas-miss weeks: 2024 wk2 (56%), wk3 (50%), 2025 wk5 (64%), wk14 and wk19 (50%). The spread-only model benefits from these just as much as the candidate.
- In 2025 wk1-4 the candidate lost to production in 3 of 4 weeks, so the early-season blend did not visibly help where production was weakest.
- Versus spread-only, weekly results are close to a coin flip but lean slightly the wrong way (17/44 wins).
- The 2026 wk1-6 gap (-0.023 vs production) comes mostly from production being near coin-flip there (LL 0.6865 vs 0.6636 for the spread alone). Versus the spread the candidate is level (dLL -0.00007).

## Interpretation
- The candidate is essentially the spread model: its probabilities correlate 0.998 with the spread-only LR (max abs difference 0.049) and its picks equal the Vegas favorite in every game. Its accuracy edge over production is just "pick the favorite".
- Against production, point estimates favor the candidate slightly (log loss -0.0023, Brier -0.0008 over 570 games), but all CIs include zero, it wins only 21/44 weeks, and dropping the best and worst weeks flips the log-loss sign. This is not distinguishable from production.
- The candidate is marginally worse than plain spread-only (log loss +0.0021, CI [-0.0003, +0.0044]; significantly worse in weeks 5+ at +0.0029 [+0.0002, +0.0055]). That matches validation (0.6189 vs 0.6185): the extra features add nothing on holdout. 2026 wk1-6 is a tie with the spread.
- The early-season blend (K=4) shows no demonstrable benefit: in weeks 1-4 the candidate is level with spread-only (-0.0005) and won only 3/8 early weeks against production. Because the candidate sits almost on the spread, any benefit from better early-season features is muted by design, so this test cannot say whether the blend itself is good.
- Nothing wrong found in the process: game ids match and the walk-forward trains strictly before each test week. Things to keep in mind: (a) the test is small (570 games; the CI half-width of about 0.013 log loss swamps the effects); (b) the main working tree has uncommitted feature changes; (c) the 2026 result largely reflects production's poor early-2026 performance; (d) this data does not show the candidate adds value over "just use the spread".

## Recommendation
Do not promote on performance grounds. There is no statistically supported improvement over production, and the candidate does not beat the trivial spread-only baseline (slightly worse, significantly so in weeks 5+). It looks like a low-risk swap (upper CI bound vs production is +0.011 log loss) but with no demonstrated upside. If a simpler, spread-anchored model is wanted, plain spread-only LR does as well or better. The final decision belongs to the user. No production or candidate files were modified.

Scripts: `bt.py` and `an.py` in the session scratchpad.
