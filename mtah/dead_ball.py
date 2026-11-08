"""Live / dead ball detector.

Entry point: detect_dead_ball.

Signals, OR-ed per frame, then cleaned of very short runs:
  1. ball tracked beyond a touchline or goal line;
  2. collective stoppage: players' median speed low for >= stoppage_min_duration, unless the
     ball is visibly moving fast;
  3. Smart Tagging set piece: dead from the tag opening until the restart kick;
  4. each half: dead from the first graded frame until the kick-off restart (the 2nd half is
     graded ~2 minutes before it kicks off).
An untracked ball is deliberately not a dead-ball signal on its own: the tracker loses the ball
in long live stretches too. It correlates with stoppages and is a candidate soft signal for a
probabilistic decoder later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import config, domain, runs


def detect_dead_ball(frame_features: domain.FrameFeatures, tagging: pd.DataFrame,
                     graded_halves: list[domain.GradedHalf],
                     pipeline_config: config.PipelineConfig) -> domain.BoolPerFrame:
    set_pieces = tagging[tagging["code"] == "set pieces"]
    set_piece_starts = set_pieces["start_video_frame"].to_numpy(dtype=np.int64)
    set_piece_restarts = np.array(
        [_find_restart_video_frame(int(start), frame_features, pipeline_config)
         for start in set_piece_starts], dtype=np.int64)

    is_dead = (frame_features.is_ball_out_of_pitch
               | _detect_collective_stoppages(frame_features, pipeline_config)
               | _detect_dead_part_of_ball_gaps(frame_features, set_piece_restarts,
                                                pipeline_config))

    for tag_start_video_frame, restart_video_frame in zip(set_piece_starts, set_piece_restarts):
        is_dead[tag_start_video_frame:restart_video_frame] = True
        # The restart kick is live even while the players are still slow around it.
        is_dead[restart_video_frame] = False

    kick_off_starts = set_pieces.loc[set_pieces["set_piece_type"] == "kick off (start)",
                                     "start_video_frame"].to_numpy()
    for graded_half in graded_halves:
        kick_off_restart_video_frame = _find_restart_video_frame(
            _first_kick_off_tag_of_half(kick_off_starts, graded_half, pipeline_config),
            frame_features, pipeline_config)
        is_dead[graded_half.first_video_frame:kick_off_restart_video_frame] = True

    is_dead = runs.drop_true_runs_shorter_than(is_dead, pipeline_config.dead_min_duration_frames)
    return runs.fill_false_runs_shorter_than(
        is_dead, pipeline_config.live_min_duration_between_dead_frames)


def _detect_collective_stoppages(frame_features: domain.FrameFeatures,
                                 pipeline_config: config.PipelineConfig) -> domain.BoolPerFrame:
    # NaN speeds compare False on both sides, so untracked frames are neither slow nor fast.
    is_ball_visibly_fast = frame_features.ball_speed_mps > pipeline_config.live_ball_min_speed_mps
    are_players_slow = (frame_features.median_player_speed_mps
                        < pipeline_config.stoppage_player_speed_mps)
    return runs.drop_true_runs_shorter_than(are_players_slow & ~is_ball_visibly_fast,
                                            pipeline_config.stoppage_min_duration_frames)


def _detect_dead_part_of_ball_gaps(frame_features: domain.FrameFeatures,
                                   set_piece_restarts: NDArray[np.int64],
                                   pipeline_config: config.PipelineConfig) -> domain.BoolPerFrame:
    """A long gap is dead from where the ball was lost until play restarts, not until the tracker
    finds the ball again: after a restart the first passes are often still untracked, and calling
    them dead starts the next possession late and can swallow short possessions whole."""
    is_dead = np.zeros(len(frame_features.is_ball_tracked), dtype=bool)
    long_gaps = [(gap_start, gap_end)
                 for gap_start, gap_end in runs.true_runs(~frame_features.is_ball_tracked)
                 if gap_end - gap_start >= pipeline_config.ball_gap_dead_min_duration_frames]
    for gap_start, gap_end in long_gaps:
        restart_video_frame = _estimate_restart_inside_gap(
            gap_start, gap_end, frame_features.median_player_speed_mps, set_piece_restarts,
            pipeline_config)
        is_dead[gap_start:restart_video_frame] = True
    return is_dead


def _estimate_restart_inside_gap(gap_start: int, gap_end: int,
                                 median_player_speed_mps: domain.FloatPerFrame,
                                 set_piece_restarts: NDArray[np.int64],
                                 pipeline_config: config.PipelineConfig) -> int:
    """The latest restart evidence inside the gap wins, because the untracked tail after the
    final restart is the live play to recover. Evidence: a set-piece restart, or the end of the
    last spell of players walking (play resuming makes everyone speed up). With neither, the
    whole gap stays dead, as before."""
    tag_restarts = set_piece_restarts[(set_piece_restarts > gap_start)
                                      & (set_piece_restarts < gap_end)]
    walking_offsets = np.flatnonzero(median_player_speed_mps[gap_start:gap_end]
                                     < pipeline_config.gap_restart_walking_speed_mps)
    restart_candidates = ([int(restart) for restart in tag_restarts]
                          + [gap_start + int(offset) + 1 for offset in walking_offsets[-1:]])
    return max(restart_candidates, default=gap_end)


def _find_restart_video_frame(tag_start_video_frame: int, frame_features: domain.FrameFeatures,
                              pipeline_config: config.PipelineConfig) -> int:
    """Tag windows open seconds before the restart, so the restart is the first fast ball
    movement in the search window; when the ball is untracked throughout, a typical offset."""
    search_end = tag_start_video_frame + pipeline_config.set_piece_restart_search_frames
    fast_offsets = np.flatnonzero(
        frame_features.ball_speed_mps[tag_start_video_frame:search_end]
        > pipeline_config.set_piece_kick_min_speed_mps)
    restart_offset = (int(fast_offsets[0]) if len(fast_offsets)
                      else pipeline_config.set_piece_default_restart_offset_frames)
    return tag_start_video_frame + restart_offset


def _first_kick_off_tag_of_half(kick_off_starts: NDArray[np.int64], graded_half: domain.GradedHalf,
                                pipeline_config: config.PipelineConfig) -> int:
    """The tag may open slightly before the half's first graded frame (the 1st half is graded from
    the kick-off itself), hence the search margin. Every half has a kick-off; a missing tag means
    the export is broken, so it raises rather than guessing."""
    earliest_allowed = (graded_half.first_video_frame
                        - pipeline_config.set_piece_restart_search_frames)
    in_half = kick_off_starts[(kick_off_starts >= earliest_allowed)
                              & (kick_off_starts <= graded_half.last_video_frame)]
    if len(in_half) == 0:
        raise ValueError("no 'Kick Off (Start)' tag for half {}".format(graded_half.half))
    return int(in_half[0])
