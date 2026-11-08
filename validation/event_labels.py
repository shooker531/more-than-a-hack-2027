"""Possession ground truth for the sample games, derived from Metrica's human-annotated events.

Entry point: derive_possession_labels.

This approximates — it is not — the challenge's ground truth, which comes from corrected
tracking. Rules, chosen to match CHALLENGE.md's definition as closely as events allow:

  Team in possession (ball alive): the team of the most recent controlled on-ball event —
    PASS, CARRY, SHOT, SET PIECE, RECOVERY, and BALL LOST (the losing team still had the ball when
    the loss began). CHALLENGE events are excluded: a duel is not control, and a contested ball
    stays with the last team in control until a RECOVERY.
  Dead ball: from the moment play stops until the next restart event —
    BALL OUT (at its end frame), a SHOT that goes out or scores, a foul (FAULT RECEIVED) that is
    followed by a SET PIECE (otherwise advantage was played), and offside. Each half is also dead
    from its first frame until the kick-off.

Known differences: event timestamps sit on touches, so boundaries are placed at touches rather
than at the moment control is established; that makes this a ranking tool between pipeline
versions, not an estimate of the challenge score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mtah import domain

_CONTROLLED_EVENT_TYPES = ("pass", "carry", "shot", "set piece", "recovery", "ball lost")
_RESTART_EVENT_TYPES = ("pass", "carry", "shot", "set piece", "recovery")


def derive_possession_labels(events: pd.DataFrame, graded_halves: list[domain.GradedHalf],
                             n_frames: int) -> domain.LabelPerFrame:
    labels = np.full(n_frames, domain.LABEL_ABSTAIN, dtype="<U1")
    for graded_half in graded_halves:
        half_events = events[events["period"] == graded_half.half]
        half_start, half_end = graded_half.first_video_frame, graded_half.last_video_frame + 1
        team_per_frame = _team_of_latest_controlled_event(half_events, half_start, half_end)
        is_dead = _dead_ball_from_events(half_events, half_start, half_end)
        labels[half_start:half_end] = np.where(
            is_dead, domain.LABEL_DEAD,
            np.where(team_per_frame == domain.HOME_TEAM, domain.LABEL_HOME, domain.LABEL_AWAY))
    return labels


def _team_of_latest_controlled_event(half_events: pd.DataFrame, half_start: int,
                                     half_end: int) -> domain.TeamCodePerFrame:
    """Frames before the first controlled event take that event's team: it is the kick-off
    taker, and those frames are dead anyway."""
    controlled = half_events[half_events["type"].isin(_CONTROLLED_EVENT_TYPES)]
    event_starts = controlled["start_video_frame"].to_numpy()
    event_teams = controlled["team"].to_numpy(dtype=np.int8)
    frames = np.arange(half_start, half_end)
    latest_event_index = np.searchsorted(event_starts, frames, side="right") - 1
    return event_teams[np.maximum(latest_event_index, 0)]


def _dead_ball_from_events(half_events: pd.DataFrame, half_start: int,
                           half_end: int) -> domain.BoolPerFrame:
    is_dead = np.zeros(half_end - half_start, dtype=bool)
    restarts = half_events[half_events["type"].isin(_RESTART_EVENT_TYPES)]
    restart_starts = restarts["start_video_frame"].to_numpy()
    restart_types = restarts["type"].to_numpy()

    kick_off_start = int(restart_starts[0])
    is_dead[:kick_off_start - half_start] = True

    for stop_video_frame, requires_set_piece in _play_stoppages(half_events):
        next_restart_index = int(np.searchsorted(restart_starts, stop_video_frame, side="right"))
        has_next_restart = next_restart_index < len(restart_starts)
        restart_video_frame = (int(restart_starts[next_restart_index]) if has_next_restart
                               else half_end)
        restarted_with_set_piece = (has_next_restart
                                    and restart_types[next_restart_index] == "set piece")
        if restarted_with_set_piece or not requires_set_piece:
            is_dead[stop_video_frame - half_start:restart_video_frame - half_start] = True
    return is_dead


def _play_stoppages(half_events: pd.DataFrame) -> list[tuple[int, bool]]:
    """(frame play stops, whether that only counts as a stoppage if a set piece follows)."""
    event_type, subtype = half_events["type"], half_events["subtype"]
    is_ball_out = event_type == "ball out"
    is_shot_ending_play = (event_type == "shot") & subtype.str.contains("out|goal", regex=True)
    is_offside = (event_type == "ball lost") & subtype.str.contains("offside")
    is_foul = event_type == "fault received"

    stoppages = [(int(frame), False) for frame in
                 half_events.loc[is_ball_out | is_shot_ending_play, "end_video_frame"]]
    stoppages += [(int(frame), False) for frame in half_events.loc[is_offside, "start_video_frame"]]
    stoppages += [(int(frame), True) for frame in half_events.loc[is_foul, "start_video_frame"]]
    return stoppages
