import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


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

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import nflreadpy as nfl

from predict_engine import load_model, build_snapshots, predict_week, log_predictions

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Loaded once at startup - a real refresh strategy comes later
model_bundle = load_model(model_path=os.path.join(os.path.dirname(__file__), "..", "models", "win_probability_model.joblib"))
snapshots = build_snapshots(
    db_path=os.path.join(os.path.dirname(__file__), "..", "data", "nfl.db"),
    seasons=range(2024, 2027)
)
teams_df = nfl.load_teams().to_pandas()
team_logos = dict(zip(teams_df['team_abbr'], teams_df['team_logo_espn']))
team_colors = dict(zip(teams_df['team_abbr'], teams_df['team_color']))
team_colors2 = dict(zip(teams_df['team_abbr'], teams_df['team_color2']))


def get_available_weeks():
    sched = nfl.load_schedules(seasons=[2026]).to_pandas()
    upcoming = sched[sched['home_score'].isna()]
    return sorted(upcoming['week'].unique().tolist())


@app.get("/")
def home(request: Request, week: int = None):
    weeks = get_available_weeks()
    if not weeks:
        return templates.TemplateResponse(
            request=request,
            name="list.html",
            context={"predictions": [], "weeks": [], "selected_week": None, "team_logos": team_logos,
                     "home_colors": {}, "away_colors": {}}
        )

    selected_week = week if week else weeks[0]
    predictions = predict_week(2026, selected_week, snapshots, model_bundle)
    log_predictions(
        predictions,
        db_path=os.path.join(os.path.dirname(__file__), "..", "data", "nfl.db"),
        season=2026, week=selected_week
    )

    home_colors, away_colors = {}, {}
    for _, g in predictions.iterrows():
        h, a = get_display_colors(g['home_team'], g['away_team'], team_colors, team_colors2)
        home_colors[g['game_id']] = h
        away_colors[g['game_id']] = a

    return templates.TemplateResponse(
        request=request,
        name="list.html",
        context={
            "predictions": predictions.to_dict('records'),
            "weeks": weeks,
            "selected_week": selected_week,
            "team_logos": team_logos,
            "home_colors": home_colors,
            "away_colors": away_colors
        }
    )

from predict_engine import get_injury_report
from predict_engine import get_live_game_data
import pandas as pd


def safe_name(table, team_col, name_col='display_name'):
    def lookup(team):
        row = table[table[team_col] == team]
        if len(row) == 0 or pd.isna(row[name_col].values[0]):
            return "Unknown"
        return row[name_col].values[0]
    return lookup


@app.get("/game/{game_id}")
def game_detail(request: Request, game_id: str, week: int):
    predictions = predict_week(2026, week, snapshots, model_bundle)
    log_predictions(
        predictions,
        db_path=os.path.join(os.path.dirname(__file__), "..", "data", "nfl.db"),
        season=2026, week=week
    )
    game_row = predictions[predictions['game_id'] == game_id]

    if game_row.empty:
        return RedirectResponse(url=f"/?week={week}")

    game = game_row.iloc[0]
    home_color, away_color = get_display_colors(game['home_team'], game['away_team'], team_colors, team_colors2)

    qb_name = safe_name(snapshots['current_qb'], 'team')
    rb_name = safe_name(snapshots['current_rb_starter'], 'team')
    wr_name = safe_name(snapshots['current_primary_wr'], 'team')

    key_players = {
        team: {"qb": qb_name(team), "rb": rb_name(team), "wr": wr_name(team)}
        for team in [game['home_team'], game['away_team']]
    }

    injuries = {
        team: get_injury_report(team, 2026, week)
        for team in [game['home_team'], game['away_team']]
    }
    live = get_live_game_data(game['home_team'], game['away_team'], game['gameday'])

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
            "live": live,
        }
    )