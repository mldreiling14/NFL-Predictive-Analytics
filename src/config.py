"""
Single source of truth for which NFL seasons this project pulls, builds
features for, and trains on. Update CURRENT_SEASON here once a year;
don't hardcode a season range anywhere else in the codebase — that's
what let 2026 silently fall out of fetch_data.py and build_features.py.
"""

FIRST_SEASON = 2015
CURRENT_SEASON = 2026

# Inclusive of both ends. Use this list directly rather than reconstructing
# a range() by hand elsewhere - range(2015, 2026) LOOKS like it includes
# 2026 but doesn't, which is exactly the bug this file exists to prevent.
SEASONS = list(range(FIRST_SEASON, CURRENT_SEASON + 1))