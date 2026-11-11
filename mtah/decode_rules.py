"""Rule-based decoder: possession claims applied in order, one live segment at a time.

Entry point: decode_possession_labels.

Inside each live segment (maximal run of not-dead frames in a graded half):
  - control and touch evidence are each merged into episodes (one team, short holes bridged);
  - the segment opens with the team of the first control or touch episode (the restart taker);
    with none, the team of a just-opened set-piece tag; failing that, the team that last held
    the ball;
  - the other team takes possession on a *claim*: a control episode of
    >= possession_switch_min_episode, or possession_switch_consecutive_touches touch episodes in
    a row with no touch by anyone else in between (a completed pass, an interception then a
    pass). A single touch or deflection makes no claim;
  - claims are applied in the order they become certain, and each switch is backdated to the
    start of its claim, because spellF1 punishes late boundaries heavily (1 s late ~ 0.65)
    while a 1-frame offset is free;
  - frames with no evidence keep the held team; a segment where no team is ever known is X.
Ungraded frames are X.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from . import config, domain, runs


class _PossessionClaim(NamedTuple):
    """Segment-relative frames. A claim becomes certain at confirmed_frame, but possession is
    dated from start_frame."""
    start_frame: int
    confirmed_frame: int
    team: int


def decode_possession_labels(control_evidence: domain.TeamCodePerFrame,
                             touch_evidence: domain.TeamCodePerFrame,
                             is_dead: domain.BoolPerFrame, tagging: pd.DataFrame,
                             graded_halves: list[domain.GradedHalf],
                             pipeline_config: config.PipelineConfig) -> domain.LabelPerFrame:
    team_per_frame = np.full(len(control_evidence), domain.NO_TEAM, dtype=np.int8)
    is_graded = np.zeros(len(control_evidence), dtype=bool)
    team_tagged_set_pieces = tagging[(tagging["code"] == "set pieces")
                                     & (tagging["team"] != domain.NO_TEAM)]

    for graded_half in graded_halves:
        half_start, half_end = graded_half.first_video_frame, graded_half.last_video_frame + 1
        is_graded[half_start:half_end] = True
        last_held_team = domain.NO_TEAM
        for live_start, live_end in runs.true_runs(~is_dead[half_start:half_end]):
            segment_start, segment_end = half_start + live_start, half_start + live_end
            control_episodes = _merge_evidence_into_episodes(
                control_evidence[segment_start:segment_end],
                pipeline_config.control_episode_max_gap_frames)
            touch_episodes = _merge_evidence_into_episodes(
                touch_evidence[segment_start:segment_end],
                pipeline_config.control_episode_max_gap_frames)
            opening_team = _choose_segment_opening_team(
                control_episodes + touch_episodes, team_tagged_set_pieces, segment_start,
                last_held_team, pipeline_config)
            claims = (_claims_from_sustained_control(control_episodes, pipeline_config)
                      + _claims_from_consecutive_touches(touch_episodes, pipeline_config))
            segment_teams = _apply_possession_claims(claims, opening_team,
                                                     segment_end - segment_start)
            team_per_frame[segment_start:segment_end] = segment_teams
            last_held_team = int(segment_teams[-1])

    return _team_codes_to_labels(team_per_frame, is_dead, is_graded)


def _merge_evidence_into_episodes(segment_evidence: domain.TeamCodePerFrame,
                                  max_gap_frames: int) -> list[domain.ControlEpisode]:
    """Frames are segment-relative. A hole of NO_TEAM up to max_gap_frames between two runs of
    the same team is bridged; any opposing evidence frame always ends the episode."""
    episodes: list[domain.ControlEpisode] = []
    for run_start, run_end, team in runs.equal_value_runs(segment_evidence):
        continues_previous = (team != domain.NO_TEAM and len(episodes) > 0
                              and episodes[-1].team == team
                              and run_start - episodes[-1].end_video_frame <= max_gap_frames)
        if continues_previous:
            episodes[-1] = episodes[-1]._replace(end_video_frame=run_end)
        elif team != domain.NO_TEAM:
            episodes.append(domain.ControlEpisode(run_start, run_end, team))
    return episodes


def _choose_segment_opening_team(episodes: list[domain.ControlEpisode],
                                 team_tagged_set_pieces: pd.DataFrame, segment_start: int,
                                 last_held_team: int,
                                 pipeline_config: config.PipelineConfig) -> int:
    """Priority: first visible player on the ball > set-piece tag that just opened > previous
    holder. The tag ranks below tracking because tags are automatic and their windows are
    imprecise."""
    first_episode = sorted(episodes, key=lambda episode: episode.start_video_frame)[:1]
    recent_tags = team_tagged_set_pieces[team_tagged_set_pieces["start_video_frame"].between(
        segment_start - pipeline_config.set_piece_restart_search_frames, segment_start)]
    candidates = ([episode.team for episode in first_episode]
                  + [int(team) for team in recent_tags["team"].iloc[-1:]]
                  + [last_held_team])
    return candidates[0]


def _claims_from_sustained_control(control_episodes: list[domain.ControlEpisode],
                                   pipeline_config: config.PipelineConfig
                                   ) -> list[_PossessionClaim]:
    min_frames = pipeline_config.possession_switch_min_episode_frames
    return [_PossessionClaim(episode.start_video_frame, episode.start_video_frame + min_frames,
                             episode.team)
            for episode in control_episodes
            if episode.end_video_frame - episode.start_video_frame >= min_frames]


def _claims_from_consecutive_touches(touch_episodes: list[domain.ControlEpisode],
                                     pipeline_config: config.PipelineConfig
                                     ) -> list[_PossessionClaim]:
    """Episodes alternate between teams only when someone else touches the ball, so a window of
    n consecutive same-team episodes is n touches by one team with no opponent touch between."""
    n_touches = pipeline_config.possession_switch_consecutive_touches
    windows = [touch_episodes[index:index + n_touches]
               for index in range(len(touch_episodes) - n_touches + 1)]
    return [_PossessionClaim(window[0].start_video_frame, window[-1].start_video_frame,
                             window[0].team)
            for window in windows
            if all(episode.team == window[0].team for episode in window)]


def _apply_possession_claims(claims: list[_PossessionClaim], opening_team: int,
                             segment_length: int) -> domain.TeamCodePerFrame:
    """A switch is never backdated past the previous switch: possession that has been decided
    stays decided."""
    segment_teams = np.full(segment_length, opening_team, dtype=np.int8)
    current_team, last_switch_frame = opening_team, 0
    for claim in sorted(claims, key=lambda claim: claim.confirmed_frame):
        if claim.team != current_team:
            switch_frame = max(claim.start_frame, last_switch_frame)
            segment_teams[switch_frame:] = claim.team
            current_team, last_switch_frame = claim.team, switch_frame
    return segment_teams


def _team_codes_to_labels(team_per_frame: domain.TeamCodePerFrame, is_dead: domain.BoolPerFrame,
                          is_graded: domain.BoolPerFrame) -> domain.LabelPerFrame:
    labels = np.full(len(team_per_frame), domain.LABEL_ABSTAIN, dtype="<U1")
    labels[is_graded & (team_per_frame == domain.HOME_TEAM)] = domain.LABEL_HOME
    labels[is_graded & (team_per_frame == domain.AWAY_TEAM)] = domain.LABEL_AWAY
    labels[is_graded & is_dead] = domain.LABEL_DEAD
    return labels
