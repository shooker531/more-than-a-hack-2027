"""Check that mtah.load reproduces the four official baseline files exactly.

Rebuilds nearest-player-to-ball team codes from our cached arrays and smooths them with the
starter's own smooth(), so any difference can only come from our loader. Prints mismatch counts.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_REPO_ROOT, os.path.join(_REPO_ROOT, "starter")]
import baseline  # noqa: E402
import games  # noqa: E402
from mtah import domain, load  # noqa: E402


def main() -> None:
    nearest_team_per_frame = _team_of_nearest_player_to_ball(
        load.load_tracking(games.GAME, games.DATA_DIR, os.path.join(games.DATA_DIR, "cache")))
    for baseline_name, smoothing_half_window_frames in baseline.VARIANTS:
        our_labels = baseline.smooth(nearest_team_per_frame, smoothing_half_window_frames)
        official_labels = pd.read_csv(os.path.join(
            _REPO_ROOT, "evaluation", "results", baseline_name + ".csv"))["label"].to_numpy()
        print("{:<20} frames differing: {}".format(
            baseline_name, int(np.sum(our_labels != official_labels))))


def _team_of_nearest_player_to_ball(tracking: domain.TrackingData) -> domain.TeamCodePerFrame:
    ball = tracking.ball_norm.astype(np.float64)
    players = tracking.players_norm.astype(np.float64)
    squared_distance_m2 = (((players[:, :, 0] - ball[:, None, 0]) * tracking.pitch_length_m) ** 2
                           + ((players[:, :, 1] - ball[:, None, 1]) * tracking.pitch_width_m) ** 2)
    has_answer = ~np.isnan(ball[:, 0]) & ~np.all(np.isnan(squared_distance_m2), axis=1)
    nearest_slot = np.argmin(np.nan_to_num(squared_distance_m2, nan=np.inf), axis=1)
    return np.where(has_answer, tracking.slot_team[nearest_slot], domain.NO_TEAM).astype(np.int8)


if __name__ == "__main__":
    main()
