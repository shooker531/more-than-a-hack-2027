"""Compare summary statistics of the real match with the degraded validation games.

    .venv/bin/python scripts/compare_harness_realism.py

The validation harness only ranks pipeline versions honestly if its degraded tracking behaves
like the challenge's automatic tracking. This prints the same statistics for both and flags any
that differ by more than ~a third. Uses no labels of any kind, and prints aggregates only.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
import games  # noqa: E402
from mtah import config, domain, evidence, features, load  # noqa: E402
from validation import degrade, sample_games  # noqa: E402

_SAMPLE_DATA_DIR = os.path.join(games.DATA_DIR, "external", "metrica-sample-data")
_SAMPLE_CACHE_DIR = os.path.join(games.DATA_DIR, "cache", "validation")
_SHARE_STATISTICS = ("ball tracked", "ball within 1 m of a player", "touch evidence",
                     "control evidence", "both teams within 1.5 m")


def main() -> None:
    real = _tracking_statistics(
        load.load_tracking(games.GAME, games.DATA_DIR, os.path.join(games.DATA_DIR, "cache")),
        load.load_graded_halves(games.GAME))
    samples = []
    for game_number in sample_games.SAMPLE_GAME_NUMBERS:
        sample_game = sample_games.load_sample_game(game_number, _SAMPLE_DATA_DIR,
                                                    _SAMPLE_CACHE_DIR)
        samples.append(_tracking_statistics(
            degrade.degrade_tracking(sample_game.tracking, degrade.DEFAULT_DEGRADE_CONFIG, 0),
            sample_game.graded_halves))
    print("{:34s} {:>10s} {:>26s}".format("statistic", "real", "samples (game 1/2/3)"))
    for name, real_value in real.items():
        sample_values = [sample[name] for sample in samples]
        is_far = not min(sample_values) * 0.75 <= real_value <= max(sample_values) * 1.33
        print("{:34s} {:>10s} {:>26s}{}".format(
            name, _format(name, real_value), " / ".join(_format(name, value)
                                                        for value in sample_values),
            "  <-- differs" if is_far else ""))


def _tracking_statistics(tracking: domain.TrackingData,
                         graded_halves: list[domain.GradedHalf]) -> dict[str, float]:
    is_graded = np.zeros(len(tracking.ball_norm), dtype=bool)
    for graded_half in graded_halves:
        is_graded[graded_half.first_video_frame:graded_half.last_video_frame + 1] = True
    frame_features = features.compute_frame_features(tracking, config.DEFAULT_CONFIG)
    is_tracked = frame_features.is_ball_tracked & is_graded
    ball_x_m = tracking.ball_norm[:, 0].astype(np.float64) * tracking.pitch_length_m
    ball_y_m = tracking.ball_norm[:, 1].astype(np.float64) * tracking.pitch_width_m
    ball_acceleration_m = np.hypot(np.diff(ball_x_m, 2), np.diff(ball_y_m, 2))
    is_tracked_three_frames = is_tracked[2:] & is_tracked[1:-1] & is_tracked[:-2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nearest_player_m = np.fmin(frame_features.nearest_home_distance_m,
                                   frame_features.nearest_away_distance_m)[is_tracked]
    both_teams_close = ((frame_features.nearest_home_distance_m < 1.5)
                        & (frame_features.nearest_away_distance_m < 1.5))[is_tracked]
    touch = evidence.detect_touch_evidence(frame_features, config.DEFAULT_CONFIG)[is_tracked]
    control = evidence.detect_control_evidence(frame_features, config.DEFAULT_CONFIG)[is_tracked]
    return {
        "ball tracked": float(np.mean(is_tracked[is_graded])),
        "ball |acceleration| p50 (m/f^2)": float(np.nanmedian(
            ball_acceleration_m[is_tracked_three_frames])),
        "ball speed p50 (m/s)": float(np.nanmedian(frame_features.ball_speed_mps[is_tracked])),
        "nearest player to ball p10 (m)": float(np.nanpercentile(nearest_player_m, 10)),
        "ball within 1 m of a player": float(np.mean(nearest_player_m <= 1.0)),
        "both teams within 1.5 m": float(np.mean(both_teams_close)),
        "touch evidence": float(np.mean(touch != domain.NO_TEAM)),
        "control evidence": float(np.mean(control != domain.NO_TEAM)),
        "median player speed p50 (m/s)": float(np.nanmedian(
            frame_features.median_player_speed_mps[is_graded])),
    }


def _format(name: str, value: float) -> str:
    return "{:.1%}".format(value) if name in _SHARE_STATISTICS else "{:.3f}".format(value)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
