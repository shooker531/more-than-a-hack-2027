"""Aggregate profile of the tactical ATD over graded frames, used to set generic thresholds.

Prints distributions only, never rows, and reads no ground truth.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from numpy.typing import NDArray

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
import games  # noqa: E402
from mtah import config, domain, features, load, runs  # noqa: E402

_PERCENTILES = [5, 25, 50, 75, 95, 99]


def main() -> None:
    tracking = load.load_tracking(games.GAME, games.DATA_DIR,
                                  os.path.join(games.DATA_DIR, "cache"))
    frame_features = features.compute_frame_features(tracking, config.DEFAULT_CONFIG)
    is_graded = np.zeros(len(tracking.ball_norm), dtype=bool)
    for graded_half in load.load_graded_halves(games.GAME):
        is_graded[graded_half.first_video_frame:graded_half.last_video_frame + 1] = True
    n_graded = int(is_graded.sum())

    is_tracked = frame_features.is_ball_tracked & is_graded
    print("graded frames {}  ball tracked {:.1%}".format(n_graded, is_tracked.sum() / n_graded))
    print("ball out of pitch (margin {}): {:.2%} of tracked frames".format(
        config.DEFAULT_CONFIG.out_of_pitch_margin_norm,
        frame_features.is_ball_out_of_pitch[is_tracked].mean()))

    gap_lengths = _run_lengths(is_graded & ~frame_features.is_ball_tracked)
    print("ball gap lengths (frames):", _describe(gap_lengths))
    for min_seconds in (1, 5, 10, 30):
        long_gaps = gap_lengths[gap_lengths >= min_seconds * config.FRAMES_PER_SECOND]
        print("  gaps >= {:>2}s: {:>4} covering {:.1%} of graded".format(
            min_seconds, len(long_gaps), long_gaps.sum() / n_graded))

    print("ball speed from positions (m/s):",
          _describe(frame_features.ball_speed_mps[is_graded]))
    print("median player speed, 1 s mean (m/s):",
          _describe(frame_features.median_player_speed_mps[is_graded]))
    for team, team_name in ((domain.HOME_TEAM, "home"), (domain.AWAY_TEAM, "away")):
        team_x_norm = tracking.players_norm[is_graded][:, tracking.slot_team == team, 0]
        tracked_count = np.sum(~np.isnan(team_x_norm), axis=1).astype(np.float64)
        print("{} players tracked per frame:".format(team_name), _describe(tracked_count))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nearest_any_m = np.fmin(frame_features.nearest_home_distance_m,
                                frame_features.nearest_away_distance_m)
    print("nearest player to ball (m):", _describe(nearest_any_m[is_tracked]))


def _run_lengths(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    return np.array([end - start for start, end in runs.true_runs(mask)], dtype=np.float64)


def _describe(values: NDArray[np.float64]) -> str:
    finite_values = values[~np.isnan(values)]
    quantiles = np.percentile(finite_values, _PERCENTILES)
    return " ".join("p{}={:.2f}".format(p, q) for p, q in zip(_PERCENTILES, quantiles)) + \
        "  (n={})".format(len(finite_values))


if __name__ == "__main__":
    main()
