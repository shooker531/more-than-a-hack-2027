"""The official baseline (nearest player to the ball, centred majority vote), on any tracking.

Entry points: team_of_nearest_player_to_ball, label_with_official_baseline.

Scored next to our pipeline on the validation games so every iteration is compared against the
published floor under identical conditions. The smoothing is the starter's own function, so this
is the official algorithm, not a reimplementation (scripts/check_baseline_repro.py proves it
reproduces the committed baseline files exactly).
"""

from __future__ import annotations

import os
import sys

import numpy as np

from mtah import domain

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "starter"))
import baseline  # noqa: E402

# The published floor is baseline_smooth50: a vote over +/- 50 frames.
_OFFICIAL_FLOOR_HALF_WINDOW_FRAMES = 50


def team_of_nearest_player_to_ball(tracking: domain.TrackingData) -> domain.TeamCodePerFrame:
    """NO_TEAM where the ball, or every player, is untracked — the baseline's 'no answer'."""
    ball = tracking.ball_norm.astype(np.float64)
    players = tracking.players_norm.astype(np.float64)
    squared_distance_m2 = (((players[:, :, 0] - ball[:, None, 0]) * tracking.pitch_length_m) ** 2
                           + ((players[:, :, 1] - ball[:, None, 1]) * tracking.pitch_width_m) ** 2)
    has_answer = ~np.isnan(ball[:, 0]) & ~np.all(np.isnan(squared_distance_m2), axis=1)
    nearest_slot = np.argmin(np.nan_to_num(squared_distance_m2, nan=np.inf), axis=1)
    return np.where(has_answer, tracking.slot_team[nearest_slot], domain.NO_TEAM).astype(np.int8)


def label_with_official_baseline(tracking: domain.TrackingData) -> domain.LabelPerFrame:
    labels: domain.LabelPerFrame = baseline.smooth(team_of_nearest_player_to_ball(tracking),
                                                   _OFFICIAL_FLOOR_HALF_WINDOW_FRAMES)
    return labels
