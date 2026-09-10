import nflreadpy as nfl


def get_injury_report(team, season, week):
    """
    Attempts to pull real injury designations for a team for a specific
    season/week.

    Uses a direct-downloader fallback because nflreadpy's own
    load_injuries() validates the requested season against
    get_current_season(), which has a real, narrow bug: it assumes
    the NFL season always starts on the "Thursday following Labor
    Day," so if a season's actual opener falls even one day earlier
    (e.g. a Wednesday opener, as in 2026), the library incorrectly
    believes the new season hasn't started yet and rejects valid
    requests with a ValueError - even though real data already
    exists on nflverse's servers (confirmed directly on the night of
    the 2026 season opener: real Week 1 Out designations were already
    published).

    Returns None only if the direct fetch itself genuinely fails, or
    a list (possibly empty) of real designations.
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
            return None

    team_injuries = injuries[
        (injuries['team'] == team) & (injuries['week'] == week) &
        (injuries['report_status'].isin(['Out', 'Doubtful', 'Questionable']))
    ]
    if team_injuries.empty:
        return []
    return team_injuries[['full_name', 'position', 'report_status']].to_dict('records')