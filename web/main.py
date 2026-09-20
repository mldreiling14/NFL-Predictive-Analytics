import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import nflreadpy as nfl
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from predict_engine import (
    load_model, build_snapshots, predict_week,
    get_injury_report, get_live_game_data, get_live_win_probability,
    log_predictions, get_logged_prediction_for_game,
)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "nfl.db")


# --- Color helpers -----------------------------------------------------

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def color_distance(hex1, hex2):
    r1, g1, b1 = hex_to_rgb(hex1)
    r2, g2, b2 = hex_to_rgb(hex2)
    return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5


def get_display_colors(home_team, away_team, colors, colors2):
    home_opts = [colors.get(home_team, '#888888'), colors2.get(home_team, '#888888')]
    away_opts = [colors.get(away_team, '#888888'), colors2.get(away_team, '#888888')]
    best_pair, best_score = (home_opts[0], away_opts[0]), -1
    for h in home_opts:
        for a in away_opts:
            dist = color_distance(h, a)
            if dist > best_score:
                best_score, best_pair = dist, (h, a)
    return best_pair


def safe_name(table, team_col, name_col='display_name'):
    def lookup(team):
        row = table[table[team_col] == team]
        if len(row) == 0 or pd.isna(row[name_col].values[0]):
            return "Unknown"
        return row[name_col].values[0]
    return lookup


# --- App setup -----------------------------------------------------------

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Loaded once at startup - a real refresh strategy comes later
model_bundle = load_model(model_path=os.path.join(os.path.dirname(__file__), "..", "models", "win_probability_model.joblib"))
snapshots = build_snapshots(db_path=DB_PATH, seasons=range(2024, 2027))

teams_df = nfl.load_teams().to_pandas()
team_logos = dict(zip(teams_df['team_abbr'], teams_df['team_logo_espn']))
team_colors = dict(zip(teams_df['team_abbr'], teams_df['team_color']))
team_colors2 = dict(zip(teams_df['team_abbr'], teams_df['team_color2']))


def get_available_weeks():
    sched = nfl.load_schedules(seasons=[2026]).to_pandas()
    upcoming = sched[sched['home_score'].isna()]
    return sorted(upcoming['week'].unique().tolist())


# --- Caching ---------------------------------------------------------------
# Predictions don't change within a week since snapshots only rebuild on
# app restart, so there's no reason to rerun predict_week() every request.
_predictions_cache = {}   # {(season, week): DataFrame}


def get_cached_predictions(season, week):
    key = (season, week)
    if key not in _predictions_cache:
        predictions = predict_week(season, week, snapshots, model_bundle)
        log_predictions(predictions, db_path=DB_PATH, season=season, week=week)
        _predictions_cache[key] = predictions
    return _predictions_cache[key]


# ESPN gets hit at most once per game per 15s, no matter how many people
# are polling or how many routes ask for the same game's live data.
_live_cache = {}   # {game_id: (fetched_at, live_dict)}
LIVE_CACHE_TTL_SECONDS = 15


def get_cached_live_data(game_id, home_team, away_team, gameday):
    now = time.time()
    cached = _live_cache.get(game_id)
    if cached and (now - cached[0]) < LIVE_CACHE_TTL_SECONDS:
        return cached[1]
    live = get_live_game_data(home_team, away_team, gameday)
    _live_cache[game_id] = (now, live)
    return live


# Injury designations update a few times a day (not second to second), so a
# 30-minute TTL keeps the detail page fast without serving a stale-by-days
# report. Cached per team/season/week, mirroring the _live_cache pattern
# above; also lets the detail page show when the report was actually pulled.
_injury_cache = {}   # {(team, season, week): (fetched_at, injuries)}
INJURY_CACHE_TTL_SECONDS = 30 * 60


def get_cached_injury_report(team, season, week):
    now = time.time()
    key = (team, season, week)
    cached = _injury_cache.get(key)
    if cached and (now - cached[0]) < INJURY_CACHE_TTL_SECONDS:
        return cached
    result = get_injury_report(team, season, week)
    _injury_cache[key] = (now, result)
    return _injury_cache[key]


def get_pregame_prediction(game_id, game_row):
    """Reads the frozen pre-game prediction from the log, falling back to
    the in-memory value only if it somehow hasn't been logged yet."""
    pregame = get_logged_prediction_for_game(game_id, db_path=DB_PATH)
    if pregame is None:
        pregame = {'home_win_prob': game_row['home_win_prob'], 'away_win_prob': game_row['away_win_prob']}
    return pregame


# --- Routes ------------------------------------------------------------

@app.get("/")
def home(request: Request, week: int = None):
    weeks = get_available_weeks()
    if not weeks:
        return templates.TemplateResponse(
            request=request,
            name="list.html",
            context={"predictions": [], "weeks": [], "selected_week": None, "team_logos": team_logos,
                     "home_colors": {}, "away_colors": {}, "live_data": {}}
        )

    selected_week = week if week else weeks[0]
    predictions = get_cached_predictions(2026, selected_week)

    home_colors, away_colors, live_data = {}, {}, {}
    for _, g in predictions.iterrows():
        h, a = get_display_colors(g['home_team'], g['away_team'], team_colors, team_colors2)
        home_colors[g['game_id']] = h
        away_colors[g['game_id']] = a
        live_data[g['game_id']] = get_cached_live_data(g['game_id'], g['home_team'], g['away_team'], g['gameday'])

    return templates.TemplateResponse(
        request=request,
        name="list.html",
        context={
            "predictions": predictions.to_dict('records'),
            "weeks": weeks,
            "selected_week": selected_week,
            "team_logos": team_logos,
            "home_colors": home_colors,
            "away_colors": away_colors,
            "live_data": live_data,
        }
    )


@app.get("/api/live/{game_id}")
def live_probability(game_id: str, week: int):
    predictions = get_cached_predictions(2026, week)
    game_row = predictions[predictions['game_id'] == game_id]
    if game_row.empty:
        return JSONResponse({"error": "not found"}, status_code=404)
    game = game_row.iloc[0]

    live = get_cached_live_data(game_id, game['home_team'], game['away_team'], game['gameday'])
    home_prob = get_live_win_probability(game['home_team'], game['away_team'], live, game['home_win_prob'])

    return {
        "state": live.get('state') if live else None,
        "home_score": live.get('home_score') if live else None,
        "away_score": live.get('away_score') if live else None,
        "quarter": live.get('quarter') if live else None,
        "clock": live.get('clock') if live else None,
        "home_win_prob": home_prob,
        "away_win_prob": 1 - home_prob,
    }


@app.get("/game/{game_id}")
def game_detail(request: Request, game_id: str, week: int):
    predictions = get_cached_predictions(2026, week)
    game_row = predictions[predictions['game_id'] == game_id]

    if game_row.empty:
        return RedirectResponse(url=f"/?week={week}")

    game = game_row.iloc[0]
    home_color, away_color = get_display_colors(game['home_team'], game['away_team'], team_colors, team_colors2)
    pregame = get_pregame_prediction(game_id, game)

    qb_name = safe_name(snapshots['current_qb'], 'team')
    rb_name = safe_name(snapshots['current_rb_starter'], 'team')
    wr_name = safe_name(snapshots['current_primary_wr'], 'team')

    key_players = {
        team: {"qb": qb_name(team), "rb": rb_name(team), "wr": wr_name(team)}
        for team in [game['home_team'], game['away_team']]
    }

    injuries = {}
    injury_fetched_at = None
    for team in [game['home_team'], game['away_team']]:
        fetched_at, report = get_cached_injury_report(team, 2026, week)
        injuries[team] = report
        if injury_fetched_at is None or fetched_at < injury_fetched_at:
            injury_fetched_at = fetched_at

    injury_fetched_at_display = datetime.fromtimestamp(
        injury_fetched_at, tz=ZoneInfo("America/New_York")
    ).strftime('%I:%M %p ET, %B %d').lstrip('0')

    live = get_cached_live_data(game_id, game['home_team'], game['away_team'], game['gameday'])
    home_prob = get_live_win_probability(game['home_team'], game['away_team'], live, game['home_win_prob'])
    return templates.TemplateResponse(
        request=request, name="detail.html",
        context={
            "game": game.to_dict(),
            "week": week,
            "gameday_display": pd.to_datetime(game['gameday']).strftime('%A, %B %d, %Y'),
            "team_logos": team_logos,
            "home_color": home_color,
            "away_color": away_color,
            "key_players": key_players,
            "injuries": injuries,
            "injury_fetched_at": injury_fetched_at_display,
            "live": live,
            "pregame_home_prob": pregame['home_win_prob'],
            "pregame_away_prob": pregame['away_win_prob'],
            "display_home_prob": home_prob,
            "display_away_prob": 1 - home_prob,
        }
    )