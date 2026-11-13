"""Per-frame features for the learned possession model: ball evidence plus team structure and
movement.

Entry points: FEATURE_NAMES, compute_model_features.

Every feature is home-minus-away (or signed so that positive favours home), so one model serves
both teams and cannot learn "home usually has the ball" from these three sample games. Team
positions are expressed in each team's attacking direction, inferred per half from where each
team's players stand on average (a team defends the half it mostly occupies) — no labels needed.

Missing values stay NaN; the gradient-boosted model handles them natively.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from . import config, domain, features

# Centred averaging windows: immediate actions, a passage of play, team structure.
_CONTEXT_WINDOWS_FRAMES = (13, 51, 125)
_PRESSING_RADIUS_M = 5.0
_CENTROID_VELOCITY_SPAN_FRAMES = 25

_INSTANT_FEATURES = (
    "nearest_home_m", "nearest_away_m", "nearest_difference_m", "ball_speed_mps",
    "is_ball_tracked", "control_evidence", "touch_evidence",
    "advance_difference_m", "forward_velocity_difference_mps", "width_difference_m",
    "length_difference_m", "speed_difference_mps", "pressing_difference",
)
_SMOOTHED_FEATURES = ("nearest_difference_m", "control_evidence", "touch_evidence",
                      "advance_difference_m", "forward_velocity_difference_mps",
                      "width_difference_m", "pressing_difference")
FEATURE_NAMES = (list(_INSTANT_FEATURES)
                 + ["{}_mean_{}f".format(name, window) for window in _CONTEXT_WINDOWS_FRAMES
                    for name in _SMOOTHED_FEATURES])


def compute_model_features(tracking: domain.TrackingData, frame_features: domain.FrameFeatures,
                           control_evidence: domain.TeamCodePerFrame,
                           touch_evidence: domain.TeamCodePerFrame,
                           graded_halves: list[domain.GradedHalf]) -> NDArray[np.float32]:
    """(n_frames, len(FEATURE_NAMES)) float32, NaN where a feature is undefined."""
    players_x_m = tracking.players_norm[:, :, 0].astype(np.float64) * tracking.pitch_length_m
    players_y_m = tracking.players_norm[:, :, 1].astype(np.float64) * tracking.pitch_width_m
    players_speed = tracking.players_norm[:, :, 2].astype(np.float64)
    is_home = tracking.slot_team == domain.HOME_TEAM
    attack_sign = _home_attack_sign_per_frame(players_x_m, is_home, graded_halves)

    ball_x_m = tracking.ball_norm[:, 0].astype(np.float64) * tracking.pitch_length_m
    ball_y_m = tracking.ball_norm[:, 1].astype(np.float64) * tracking.pitch_width_m
    distance_to_ball_m = np.hypot(players_x_m - ball_x_m[:, None], players_y_m - ball_y_m[:, None])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_team = {team: _team_shape(players_x_m[:, mask], players_y_m[:, mask],
                                      players_speed[:, mask], distance_to_ball_m[:, mask],
                                      sign, tracking.pitch_length_m)
                    for team, mask, sign in ((domain.HOME_TEAM, is_home, attack_sign),
                                             (domain.AWAY_TEAM, ~is_home, -attack_sign))}

    home, away = per_team[domain.HOME_TEAM], per_team[domain.AWAY_TEAM]
    instant = {
        "nearest_home_m": frame_features.nearest_home_distance_m,
        "nearest_away_m": frame_features.nearest_away_distance_m,
        "nearest_difference_m": (frame_features.nearest_away_distance_m
                                 - frame_features.nearest_home_distance_m),
        "ball_speed_mps": frame_features.ball_speed_mps,
        "is_ball_tracked": frame_features.is_ball_tracked.astype(np.float64),
        "control_evidence": _signed_team_evidence(control_evidence),
        "touch_evidence": _signed_team_evidence(touch_evidence),
        "advance_difference_m": home["advance_m"] - away["advance_m"],
        "forward_velocity_difference_mps": home["forward_velocity_mps"]
        - away["forward_velocity_mps"],
        "width_difference_m": home["width_m"] - away["width_m"],
        "length_difference_m": home["length_m"] - away["length_m"],
        "speed_difference_mps": home["mean_speed_mps"] - away["mean_speed_mps"],
        "pressing_difference": home["players_near_ball"] - away["players_near_ball"],
    }
    columns = [instant[name] for name in _INSTANT_FEATURES]
    for window in _CONTEXT_WINDOWS_FRAMES:
        columns += [features.centred_nan_mean(instant[name], window) for name in _SMOOTHED_FEATURES]
    return np.column_stack(columns).astype(np.float32)


def _home_attack_sign_per_frame(players_x_m: NDArray[np.float64], is_home: NDArray[np.bool_],
                                graded_halves: list[domain.GradedHalf]) -> domain.FloatPerFrame:
    """+1 where home attacks towards increasing x, -1 otherwise. A team spends most of a half in
    its own defensive half, so the team with the lower average x defends the low-x goal."""
    sign = np.ones(len(players_x_m))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for graded_half in graded_halves:
            half = slice(graded_half.first_video_frame, graded_half.last_video_frame + 1)
            home_mean_x = np.nanmean(players_x_m[half][:, is_home])
            away_mean_x = np.nanmean(players_x_m[half][:, ~is_home])
            sign[half] = 1.0 if home_mean_x < away_mean_x else -1.0
    return sign



def _team_shape(team_x_m: NDArray[np.float64], team_y_m: NDArray[np.float64],
                team_speed: NDArray[np.float64], team_distance_to_ball_m: NDArray[np.float64],
                attack_sign: NDArray[np.float64], pitch_length_m: float
                ) -> dict[str, domain.FloatPerFrame]:
    """advance_m: team centroid's distance from its own goal line. The team in possession pushes
    up, spreads wide and long, and moves forward; the defending team drops and compacts."""
    attacking_x_m = np.where(attack_sign[:, None] > 0, team_x_m, pitch_length_m - team_x_m)
    advance_m = np.nanmean(attacking_x_m, axis=1)
    span = _CENTROID_VELOCITY_SPAN_FRAMES
    forward_velocity = np.full(len(advance_m), np.nan)
    forward_velocity[span:] = ((advance_m[span:] - advance_m[:-span])
                               * config.FRAMES_PER_SECOND / span)
    return {
        "advance_m": advance_m,
        "forward_velocity_mps": forward_velocity,
        "width_m": np.nanstd(team_y_m, axis=1),
        "length_m": np.nanstd(attacking_x_m, axis=1),
        "mean_speed_mps": np.nanmean(team_speed, axis=1),
        "players_near_ball": np.sum(team_distance_to_ball_m <= _PRESSING_RADIUS_M,
                                    axis=1).astype(np.float64),
    }


def _signed_team_evidence(evidence: domain.TeamCodePerFrame) -> domain.FloatPerFrame:
    return np.where(evidence == domain.HOME_TEAM, 1.0,
                    np.where(evidence == domain.AWAY_TEAM, -1.0, 0.0))

