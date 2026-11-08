"""Per-frame control evidence: which team, if any, visibly controls the ball on this frame.

Entry point: detect_control_evidence.

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
