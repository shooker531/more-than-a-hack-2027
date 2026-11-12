"""Per-frame evidence of which team, if any, is on the ball.

Entry points: detect_control_evidence, detect_touch_evidence.

Evidence is deliberately sparse. Untracked balls, loose balls and contested balls give NO_TEAM,
and the decoder holds the previous team through them (CHALLENGE.md: a contested or loose ball
stays with the last team in possession).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import config, domain


def detect_control_evidence(frame_features: domain.FrameFeatures,
                            pipeline_config: config.PipelineConfig) -> domain.TeamCodePerFrame:
    """A slow ball (not a pass in flight) with a player close to it."""
    is_ball_in_flight = (frame_features.ball_speed_mps
                         > pipeline_config.control_max_ball_speed_mps)
    return _team_clearly_on_the_ball(frame_features,
                                     frame_features.is_ball_tracked & ~is_ball_in_flight,
                                     pipeline_config.control_radius_m,
                                     pipeline_config.control_contest_margin_m)


def detect_touch_evidence(frame_features: domain.FrameFeatures,
                          pipeline_config: config.PipelineConfig) -> domain.TeamCodePerFrame:
    """A player close enough to be playing the ball, at any ball speed. Control evidence ignores
    fast balls, which makes it blind at turnovers: on the sample games, when a true possession
    showed no control evidence, the possessing team's nearest player was a median 0.4 m from the
    ball while it moved at a median 11.9 m/s (interceptions, first-time passes). One touch is
    not possession (a deflection, a 50/50); the decoder decides what a run of touches means."""
    return _team_clearly_on_the_ball(frame_features, frame_features.is_ball_tracked,
                                     pipeline_config.touch_radius_m,
                                     pipeline_config.touch_contest_margin_m)


def _team_clearly_on_the_ball(frame_features: domain.FrameFeatures,
                              is_eligible: NDArray[np.bool_], radius_m: float,
                              contest_margin_m: float) -> domain.TeamCodePerFrame:
    """The team whose nearest player is within radius_m AND at least contest_margin_m nearer than
    any opponent. In a duel or under tight marking both players are close, and the nearer one is
    as often the defender as the ball carrier, so neither team gets evidence. Untracked teams are
    +inf, so they are never the nearer team."""
    nearest_home_m = np.nan_to_num(frame_features.nearest_home_distance_m, nan=np.inf)
    nearest_away_m = np.nan_to_num(frame_features.nearest_away_distance_m, nan=np.inf)
    is_home_clear = (is_eligible & (nearest_home_m <= radius_m)
                     & (nearest_home_m + contest_margin_m <= nearest_away_m))
    is_away_clear = (is_eligible & (nearest_away_m <= radius_m)
                     & (nearest_away_m + contest_margin_m <= nearest_home_m))

    evidence = np.full(len(nearest_home_m), domain.NO_TEAM, dtype=np.int8)
    evidence[is_home_clear] = domain.HOME_TEAM
    evidence[is_away_clear] = domain.AWAY_TEAM
    return evidence
