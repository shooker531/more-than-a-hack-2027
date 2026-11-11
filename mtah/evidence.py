"""Per-frame control evidence: which team, if any, visibly controls the ball on this frame.

Entry points: detect_control_evidence, detect_touch_evidence.

Evidence is deliberately sparse: only a slow, tracked ball with a player close to it counts.
Passes in flight, untracked balls and loose balls give NO_TEAM, and the decoder holds the
previous team through them (loose balls stay with the last team in control, per CHALLENGE.md).
"""

from __future__ import annotations

import numpy as np

from . import config, domain


def detect_control_evidence(frame_features: domain.FrameFeatures,
                            pipeline_config: config.PipelineConfig) -> domain.TeamCodePerFrame:
    # Untracked teams become +inf so they can never be the nearer team.
    nearest_home_m = np.nan_to_num(frame_features.nearest_home_distance_m, nan=np.inf)
    nearest_away_m = np.nan_to_num(frame_features.nearest_away_distance_m, nan=np.inf)
    is_ball_in_flight = (frame_features.ball_speed_mps
                         > pipeline_config.control_max_ball_speed_mps)
    is_controlled = (frame_features.is_ball_tracked & ~is_ball_in_flight
                     & (np.minimum(nearest_home_m, nearest_away_m)
                        <= pipeline_config.control_radius_m))

    control_evidence = np.full(len(nearest_home_m), domain.NO_TEAM, dtype=np.int8)
    control_evidence[is_controlled & (nearest_home_m <= nearest_away_m)] = domain.HOME_TEAM
    control_evidence[is_controlled & (nearest_away_m < nearest_home_m)] = domain.AWAY_TEAM
    return control_evidence


def detect_touch_evidence(frame_features: domain.FrameFeatures,
                          pipeline_config: config.PipelineConfig) -> domain.TeamCodePerFrame:
    """A player close enough to be playing the ball, at any ball speed. Control evidence ignores
    fast balls, which makes it blind at turnovers: on the sample games, when a true possession
    showed no control evidence, the possessing team's nearest player was a median 0.4 m from the
    ball while it moved at a median 11.9 m/s (interceptions, first-time passes). One touch is
    not possession (a deflection, a 50/50); the decoder decides what a run of touches means."""
    nearest_home_m = np.nan_to_num(frame_features.nearest_home_distance_m, nan=np.inf)
    nearest_away_m = np.nan_to_num(frame_features.nearest_away_distance_m, nan=np.inf)
    is_touching = (frame_features.is_ball_tracked
                   & (np.minimum(nearest_home_m, nearest_away_m)
                      <= pipeline_config.touch_radius_m))

    touch_evidence = np.full(len(nearest_home_m), domain.NO_TEAM, dtype=np.int8)
    touch_evidence[is_touching & (nearest_home_m <= nearest_away_m)] = domain.HOME_TEAM
    touch_evidence[is_touching & (nearest_away_m < nearest_home_m)] = domain.AWAY_TEAM
    return touch_evidence
