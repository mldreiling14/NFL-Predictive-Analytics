import nflreadpy as nfl


def get_injury_report(team, season, week):
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
            return None

    team_injuries = injuries[
        (injuries['team'] == team) & (injuries['week'] == week) &
        (injuries['report_status'].isin(['Out', 'Doubtful', 'Questionable']))
    ]
    if team_injuries.empty:
        return []
    return team_injuries[['gsis_id', 'full_name', 'position', 'report_status']].to_dict('records')

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