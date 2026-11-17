"""Shared type aliases, domain constants and the immutable records passed between pipeline steps.

Sentinels inside numeric arrays keep every array a single type:
  NO_TEAM (-1) in team-code arrays  -> no team (no evidence, or no possession decided)
  NaN in float arrays               -> not tracked on that frame
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

HOME_TEAM: int = 0
AWAY_TEAM: int = 1
NO_TEAM: int = -1

LABEL_HOME, LABEL_AWAY, LABEL_DEAD, LABEL_ABSTAIN = "H", "A", "D", "X"

BoolPerFrame: TypeAlias = NDArray[np.bool_]
FloatPerFrame: TypeAlias = NDArray[np.float64]
TeamCodePerFrame: TypeAlias = NDArray[np.int8]
LabelPerFrame: TypeAlias = NDArray[np.str_]


class GradedHalf(NamedTuple):
    """Video-frame span of one graded half; last_video_frame is inclusive (as in games.py)."""
    half: int
    first_video_frame: int
    last_video_frame: int


@dataclass(frozen=True)
class TrackingData:
    """Tactical ATD indexed by video frame (ATD frame_id - 1).

    ball_norm:    (n_frames, 3) x, y in normalised pitch units, speed m/s
    players_norm: (n_frames, n_slots, 3) per track slot, same channels
    slot_team:    (n_slots,) HOME_TEAM / AWAY_TEAM. Slots are tracklets, not people.
    """
    ball_norm: NDArray[np.float32]
    players_norm: NDArray[np.float32]
    slot_team: NDArray[np.int8]
    pitch_length_m: float
    pitch_width_m: float


@dataclass(frozen=True)
class FrameFeatures:
    is_ball_tracked: BoolPerFrame
    is_ball_out_of_pitch: BoolPerFrame
    ball_speed_mps: FloatPerFrame
    nearest_home_distance_m: FloatPerFrame
    nearest_away_distance_m: FloatPerFrame
    median_player_speed_mps: FloatPerFrame


class ControlEpisode(NamedTuple):
    """Consecutive frames of one team's control evidence; end_video_frame is exclusive."""
    start_video_frame: int
    end_video_frame: int
    team: int
