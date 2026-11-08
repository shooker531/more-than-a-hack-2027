"""Per-frame features from the tracking arrays, vectorised over all frames.

Entry point: compute_frame_features.

All distances are computed in metres: the pitch is not square, so distances in normalised units
would stretch the short axis and pick the wrong nearest player near the touchlines.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from . import config, domain


def compute_frame_features(tracking: domain.TrackingData,
                           pipeline_config: config.PipelineConfig) -> domain.FrameFeatures:
    ball_norm = tracking.ball_norm.astype(np.float64)
    players_norm = tracking.players_norm.astype(np.float64)
    ball_x_m = ball_norm[:, 0] * tracking.pitch_length_m
    ball_y_m = ball_norm[:, 1] * tracking.pitch_width_m
    player_to_ball_distance_m = np.hypot(
        players_norm[:, :, 0] * tracking.pitch_length_m - ball_x_m[:, None],
        players_norm[:, :, 1] * tracking.pitch_width_m - ball_y_m[:, None])

    with warnings.catch_warnings():
        # All-NaN frames (nobody tracked) legitimately produce NaN medians.
        warnings.simplefilter("ignore", RuntimeWarning)
        median_player_speed_mps = np.nanmedian(players_norm[:, :, 2], axis=1)

    return domain.FrameFeatures(
        is_ball_tracked=~np.isnan(ball_norm[:, 0]),
        is_ball_out_of_pitch=_is_ball_beyond_lines(ball_norm,
                                                   pipeline_config.out_of_pitch_margin_norm),
        ball_speed_mps=_ball_speed_from_positions(ball_x_m, ball_y_m,
                                                  pipeline_config.ball_speed_window_frames),
        nearest_home_distance_m=_nearest_distance_of_team(player_to_ball_distance_m,
                                                          tracking.slot_team, domain.HOME_TEAM),
        nearest_away_distance_m=_nearest_distance_of_team(player_to_ball_distance_m,
                                                          tracking.slot_team, domain.AWAY_TEAM),
        median_player_speed_mps=_centred_nan_mean(
            median_player_speed_mps, pipeline_config.player_speed_smoothing_frames),
    )


def _is_ball_beyond_lines(ball_norm: NDArray[np.float64], margin_norm: float) -> NDArray[np.bool_]:
    """NaN positions compare False, so an untracked ball is never 'out'."""
    x, y = ball_norm[:, 0], ball_norm[:, 1]
    return (x < -margin_norm) | (x > 1 + margin_norm) | (y < -margin_norm) | (y > 1 + margin_norm)


def _ball_speed_from_positions(ball_x_m: domain.FloatPerFrame, ball_y_m: domain.FloatPerFrame,
                               window_frames: int) -> domain.FloatPerFrame:
    """Displacement between frame f - w/2 and f + w/2, so single-frame jitter averages out.
    NaN at the edges and wherever either endpoint is untracked."""
    half_window = window_frames // 2
    span = 2 * half_window
    speed_mps = np.full(len(ball_x_m), np.nan)
    displacement_m = np.hypot(ball_x_m[span:] - ball_x_m[:len(ball_x_m) - span],
                              ball_y_m[span:] - ball_y_m[:len(ball_y_m) - span])
    speed_mps[half_window:half_window + len(displacement_m)] = (
        displacement_m * config.FRAMES_PER_SECOND / span)
    return speed_mps


def _nearest_distance_of_team(player_to_ball_distance_m: NDArray[np.float64],
                              slot_team: NDArray[np.int8], team: int) -> domain.FloatPerFrame:
    """NaN when the ball or every player of the team is untracked on that frame."""
    team_distances_m = player_to_ball_distance_m[:, slot_team == team]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(team_distances_m, axis=1)


def _centred_nan_mean(values: domain.FloatPerFrame, window_frames: int) -> domain.FloatPerFrame:
    """Cumulative sums keep the cost independent of the window size."""
    is_valid = ~np.isnan(values)
    cumulative_sum = np.concatenate(([0.0], np.cumsum(np.where(is_valid, values, 0.0))))
    cumulative_count = np.concatenate(([0], np.cumsum(is_valid)))
    frame_index = np.arange(len(values))
    window_start = np.maximum(0, frame_index - window_frames // 2)
    window_end = np.minimum(len(values), frame_index + window_frames // 2 + 1)
    count = cumulative_count[window_end] - cumulative_count[window_start]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0,
                        (cumulative_sum[window_end] - cumulative_sum[window_start]) / count,
                        np.nan)
