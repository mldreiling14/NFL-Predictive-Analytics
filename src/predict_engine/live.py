import requests

# ESPN uses slightly different team abbreviations than nflreadpy for a few teams.
# May need adjustment if other mismatches turn up during testing.
ESPN_ABBR_MAP = {
    'WAS': 'WSH',
    'LA': 'LAR',
}


def _to_espn_abbr(team):
    return ESPN_ABBR_MAP.get(team, team)


def get_live_game_data(home_team, away_team, gameday):
    """
    Attempts to fetch live/final score and situation data for a
    specific game from ESPN's public (unofficial) scoreboard API.

    Returns a dict describing the game's current state:
    - 'available': True + full live detail (score, quarter, clock,
      possession, down/distance) if the game is currently in progress
    - 'available': False, 'state': 'post' + final score if the game
      has ended
    - 'available': False, 'state': 'pre' if it hasn't started yet
    - 'available': False, 'state': None if the fetch itself fails
      for any reason (network issue, unexpected response shape) -
      this is an undocumented, third-party source that could change
      without notice, so this stays defensive rather than raising.
    """
    try:
        date_str = gameday.replace('-', '') if isinstance(gameday, str) else gameday.strftime('%Y%m%d')
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={date_str}"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        home_espn = _to_espn_abbr(home_team)
        away_espn = _to_espn_abbr(away_team)

        for event in data.get('events', []):
            comp = event.get('competitions', [{}])[0]
            competitors = comp.get('competitors', [])

            found_home = found_away = False
            home_score = away_score = None
            for c in competitors:
                abbr = c.get('team', {}).get('abbreviation', '')
                if abbr == home_espn:
                    found_home = True
                    home_score = c.get('score')
                elif abbr == away_espn:
                    found_away = True
                    away_score = c.get('score')

            if not (found_home and found_away):
                continue

            status = comp.get('status', {})
            state = status.get('type', {}).get('state')  # 'pre', 'in', 'post'

            if state == 'post':
                return {
                    'available': False,
                    'state': 'post',
                    'home_score': home_score,
                    'away_score': away_score,
                }

            if state != 'in':
                return {'available': False, 'state': state}

            situation = comp.get('situation', {})
            possession_id = situation.get('possession')
            possession_team = None
            for c in competitors:
                if c.get('id') == possession_id:
                    possession_team = c.get('team', {}).get('abbreviation')

            down_distance = situation.get('shortDownDistanceText')

            return {
                'available': True,
                'state': 'in',
                'quarter': status.get('period'),
                'clock': status.get('displayClock'),
                'home_score': home_score,
                'away_score': away_score,
                'possession_team': possession_team,
                'down_distance': down_distance,
            }

        return {'available': False, 'state': None}

    except Exception as e:
        print(f"Live data fetch failed: {e}")
        return {'available': False, 'state': None}