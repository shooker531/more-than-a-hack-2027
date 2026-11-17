"""Possession ground truth for the sample games, derived from Metrica's human-annotated events.

Entry point: derive_possession_labels.

This approximates — it is not — the challenge's ground truth, which comes from corrected
tracking. Rules, chosen to match CHALLENGE.md's definition as closely as events allow:

  Team in possession (ball alive): the team of the most recent on-ball event —
    PASS, CARRY, SHOT, SET PIECE, RECOVERY, and BALL LOST (the losing team still had the ball when
    the loss began). CHALLENGE events are excluded: a duel is not control, and a contested ball
    stays with the last team in control until a RECOVERY.
  Controlled possession only: a new team's spell counts only if it shows control — a completed
    pass (Metrica's PASS is always completed), a carry, a shot, a set piece, or keeping the ball
    for >= 2 s of live play (CHALLENGE.md: "keeps the ball for a few seconds"). A ball won and
    lost straight away is merged into the previous team's possession, as CHALLENGE.md says a
    touch without control is not a change of possession.
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

from mtah import domain, runs

_CONTROLLED_EVENT_TYPES = ("pass", "carry", "shot", "set piece", "recovery", "ball lost")
_RESTART_EVENT_TYPES = ("pass", "carry", "shot", "set piece", "recovery")
_CONTROL_SHOWING_EVENT_TYPES = ("pass", "carry", "shot", "set piece")
_MIN_CONTROLLED_SPELL_FRAMES = 2 * 25


def derive_possession_labels(events: pd.DataFrame, graded_halves: list[domain.GradedHalf],
                             n_frames: int) -> domain.LabelPerFrame:
    labels = np.full(n_frames, domain.LABEL_ABSTAIN, dtype="<U1")
    for graded_half in graded_halves:
        half_events = events[events["period"] == graded_half.half]
        half_start, half_end = graded_half.first_video_frame, graded_half.last_video_frame + 1
        is_dead = _dead_ball_from_events(half_events, half_start, half_end)
        team_per_frame = _team_of_latest_controlled_event(half_events, half_start, half_end,
                                                          is_dead)
        labels[half_start:half_end] = np.where(
            is_dead, domain.LABEL_DEAD,
            np.where(team_per_frame == domain.HOME_TEAM, domain.LABEL_HOME, domain.LABEL_AWAY))
    return labels


def _team_of_latest_controlled_event(half_events: pd.DataFrame, half_start: int,
                                     half_end: int,
                                     is_dead: domain.BoolPerFrame) -> domain.TeamCodePerFrame:
    """Frames before the first controlled event take that event's team: it is the kick-off
    taker, and those frames are dead anyway."""
    controlled = half_events[half_events["type"].isin(_CONTROLLED_EVENT_TYPES)]
    event_starts = controlled["start_video_frame"].to_numpy()
    event_teams = _merge_spells_without_control(
        event_starts, controlled["team"].to_numpy(dtype=np.int8),
        controlled["type"].to_numpy(), half_start, half_end, is_dead)
    frames = np.arange(half_start, half_end)
    latest_event_index = np.searchsorted(event_starts, frames, side="right") - 1
    return event_teams[np.maximum(latest_event_index, 0)]


def _merge_spells_without_control(event_starts: np.ndarray, event_teams: domain.TeamCodePerFrame,
                                  event_types: np.ndarray, half_start: int, half_end: int,
                                  is_dead: domain.BoolPerFrame) -> domain.TeamCodePerFrame:
    """A spell is a run of consecutive events by one team. Its length counts live frames only,
    so a ball won just before it goes out of play does not count as kept."""
    merged_teams = event_teams.copy()
    spell_bounds = runs.equal_value_runs(event_teams)
    spell_ends_frame = [int(event_starts[next_start]) for _, next_start, _ in spell_bounds[:-1]]
    spell_ends_frame.append(half_end)
    held_team = int(event_teams[0])
    for (first_event, end_event, team), end_frame in zip(spell_bounds, spell_ends_frame):
        start_frame = int(event_starts[first_event])
        live_frames = int(np.sum(~is_dead[start_frame - half_start:end_frame - half_start]))
        shows_control = (np.isin(event_types[first_event:end_event],
                                 _CONTROL_SHOWING_EVENT_TYPES).any()
                         or live_frames >= _MIN_CONTROLLED_SPELL_FRAMES)
        held_team = team if shows_control else held_team
        merged_teams[first_event:end_event] = held_team
    return merged_teams


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
