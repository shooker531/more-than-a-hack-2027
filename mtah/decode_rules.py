"""MVP decoder: a hysteresis state machine over control episodes, one live segment at a time.

Entry point: decode_possession_labels.

Inside each live segment (maximal run of not-dead frames in a graded half):
  - control evidence is merged into episodes (one team, short holes bridged);
  - the segment opens with the team of the first episode (the restart taker); with no episode,
    the team of a just-opened set-piece tag; failing that, the team that last held the ball;
  - possession switches only on an opposing episode of >= possession_switch_min_episode. A touch
    or deflection is shorter, so it does not flip possession;
  - a switch is backdated to the start of the winning episode, because spellF1 punishes late
    boundaries heavily (1 s late ~ 0.65) while a 1-frame offset is free;
  - frames with no evidence keep the held team; a segment where no team is ever known is X.
Ungraded frames are X.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, domain, runs


def decode_possession_labels(control_evidence: domain.TeamCodePerFrame,
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
            episodes = _merge_evidence_into_control_episodes(
                control_evidence[segment_start:segment_end],
                pipeline_config.control_episode_max_gap_frames)
            opening_team = _choose_segment_opening_team(
                episodes, team_tagged_set_pieces, segment_start, last_held_team, pipeline_config)
            segment_teams = _apply_possession_switches(
                episodes, opening_team, segment_end - segment_start, pipeline_config)
            team_per_frame[segment_start:segment_end] = segment_teams
            last_held_team = int(segment_teams[-1])

    return _team_codes_to_labels(team_per_frame, is_dead, is_graded)


def _merge_evidence_into_control_episodes(segment_evidence: domain.TeamCodePerFrame,
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
    """Priority: first visible controller > set-piece tag that just opened > previous holder.
    The tag ranks below tracking because tags are automatic and their windows are imprecise."""
    recent_tags = team_tagged_set_pieces[team_tagged_set_pieces["start_video_frame"].between(
        segment_start - pipeline_config.set_piece_restart_search_frames, segment_start)]
    candidates = ([episode.team for episode in episodes[:1]]
                  + [int(team) for team in recent_tags["team"].iloc[-1:]]
                  + [last_held_team])
    return candidates[0]


def _apply_possession_switches(episodes: list[domain.ControlEpisode], opening_team: int,
                               segment_length: int,
                               pipeline_config: config.PipelineConfig) -> domain.TeamCodePerFrame:
    segment_teams = np.full(segment_length, opening_team, dtype=np.int8)
    current_team = opening_team
    for episode in episodes:
        episode_length = episode.end_video_frame - episode.start_video_frame
        takes_possession = (episode.team != current_team
                            and episode_length
                            >= pipeline_config.possession_switch_min_episode_frames)
        if takes_possession:
            segment_teams[episode.start_video_frame:] = episode.team
            current_team = episode.team
    return segment_teams


def _team_codes_to_labels(team_per_frame: domain.TeamCodePerFrame, is_dead: domain.BoolPerFrame,
                          is_graded: domain.BoolPerFrame) -> domain.LabelPerFrame:
    labels = np.full(len(team_per_frame), domain.LABEL_ABSTAIN, dtype="<U1")
    labels[is_graded & (team_per_frame == domain.HOME_TEAM)] = domain.LABEL_HOME
    labels[is_graded & (team_per_frame == domain.AWAY_TEAM)] = domain.LABEL_AWAY
    labels[is_graded & is_dead] = domain.LABEL_DEAD
    return labels
