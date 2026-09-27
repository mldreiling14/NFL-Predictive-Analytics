import nflreadpy as nfl
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

ESPN_TEAM_IDS = {
    'ARI': 22, 'ATL': 1, 'BAL': 33, 'BUF': 2, 'CAR': 29, 'CHI': 3, 'CIN': 4,
    'CLE': 5, 'DAL': 6, 'DEN': 7, 'DET': 8, 'GB': 9, 'HOU': 34, 'IND': 11,
    'JAX': 30, 'KC': 12, 'LV': 13, 'LAC': 24, 'LAR': 14, 'MIA': 15, 'MIN': 16,
    'NE': 17, 'NO': 18, 'NYG': 19, 'NYJ': 20, 'PHI': 21, 'PIT': 23, 'SF': 25,
    'SEA': 26, 'TB': 27, 'TEN': 10, 'WAS': 28,
}

# Weekly practice-report designations, plus long-term/season-ending ones -
# together, these are what actually mean "this player might not play."
ACTIVE_INJURY_STATUSES = {
    'Out', 'Doubtful', 'Questionable',
    'Injured Reserve', 'Physically Unable to Perform', 'Reserve/PUP',
}

REQUEST_TIMEOUT = 5
MAX_RECENT_ITEMS = 25  # ESPN's list is a full-season log, not just this week -
                       # this caps how many of the most recent entries we check


def _fetch_json(url):
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def get_injury_report(team, season=None, week=None):
    """
    Pulls current injury designations directly from ESPN's core API
    instead of nflreadpy, since ESPN updates same-day while nflreadpy's
    injury data lags behind actual news.

    ESPN's team injuries endpoint returns a running SEASON LOG (every
    status change all year for every player, not just this week's
    report), so this keeps only the most recent entry per player, then
    filters to designations that mean the player might be unavailable.

    `season`/`week` are accepted but unused - kept so this drops in as
    a replacement without touching every call site.
    """
    team_id = ESPN_TEAM_IDS.get(team)
    if team_id is None:
        return None

    list_url = (
        f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"
        f"teams/{team_id}/injuries?limit=100"
    )
    listing = _fetch_json(list_url)
    if not listing or not listing.get('items'):
        return []

    item_refs = [item['$ref'] for item in listing['items'][:MAX_RECENT_ITEMS]]

    details = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(_fetch_json, ref) for ref in item_refs]
        for f in as_completed(futures):
            result = f.result()
            if result:
                details.append(result)

    # Keep only the most recent entry per athlete - the log includes every
    # past status change, and an old "Questionable" shouldn't shadow a
    # newer "Out" (or a newer return to health, which just gets dropped
    # below since it won't be in ACTIVE_INJURY_STATUSES).
    latest_by_athlete = {}
    for d in details:
        athlete_ref = (d.get('athlete') or {}).get('$ref')
        if not athlete_ref:
            continue
        existing = latest_by_athlete.get(athlete_ref)
        if existing is None or d.get('date', '') > existing.get('date', ''):
            latest_by_athlete[athlete_ref] = d

    active = [d for d in latest_by_athlete.values() if d.get('status') in ACTIVE_INJURY_STATUSES]
    if not active:
        return []

    # One more hop per surviving player, to get their name/position.
    results = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        athlete_futures = {pool.submit(_fetch_json, d['athlete']['$ref']): d for d in active}
        for f in as_completed(athlete_futures):
            injury = athlete_futures[f]
            athlete = f.result()
            if not athlete:
                continue
            position = (athlete.get('position') or {}).get('abbreviation', '')
            results.append({
                'full_name': athlete.get('displayName', 'Unknown'),
                'position': position,
                'report_status': injury.get('status'),
            })

    return results


def get_injured_player_ids(season, week):
    """
    Returns the set of gsis_ids with an Out/Doubtful/Questionable
    designation for a given season/week, across all teams - a single
    league-wide fetch, reused for every team/position flag check
    rather than re-fetching per team.
    """
    try:
        injuries = nfl.load_injuries(seasons=[season]).to_pandas()
    except ValueError:
        try:
            from nflreadpy.downloader import get_downloader
            downloader = get_downloader()
            injuries = downloader.download(
                'nflverse-data', f'injuries/injuries_{season}', season=season
            ).to_pandas()
        except Exception:
            return set()

    flagged = injuries[
        (injuries['week'] == week) &
        (injuries['report_status'].isin(['Out', 'Doubtful', 'Questionable']))
    ]
    return set(flagged['gsis_id'].dropna())