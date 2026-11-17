"""Viterbi decoder: the most likely H / A sequence inside each live stretch, given the model's
per-frame probabilities.

Entry point: decode_possession_labels.

Dead ball comes from the rule-based detector (dead_ball.detect_dead_ball) and is kept as is.
Inside each live stretch, each frame scores log P(home) for H and log P(away) for A, and every
change of team pays model_switch_cost. Viterbi finds the labelling with the highest total.
So possession switches only when the model is confidently on the other side for long enough:
a few uncertain frames cannot flip it, a sustained shift does, and the switch lands where the
evidence turns rather than after a fixed confirmation delay.

An earlier version scored hand-set evidence weights (iteration 007) and lost to the rules; the
difference is calibrated probabilities from a model that learned what each cue is worth.
Ungraded frames are X.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import config, domain, runs

_HOME_STATE, _AWAY_STATE = 0, 1


def decode_possession_labels(home_probability: domain.FloatPerFrame,
                             is_dead: domain.BoolPerFrame,
                             graded_halves: list[domain.GradedHalf],
                             pipeline_config: config.PipelineConfig) -> domain.LabelPerFrame:
    floor = pipeline_config.model_probability_floor
    clipped = np.clip(home_probability, floor, 1.0 - floor)
    log_probabilities = np.column_stack((np.log(clipped), np.log(1.0 - clipped)))

    labels = np.full(len(home_probability), domain.LABEL_ABSTAIN, dtype="<U1")
    for graded_half in graded_halves:
        half_start, half_end = graded_half.first_video_frame, graded_half.last_video_frame + 1
        labels[half_start:half_end] = domain.LABEL_DEAD
        for live_start, live_end in runs.true_runs(~is_dead[half_start:half_end]):
            live = slice(half_start + live_start, half_start + live_end)
            states = _most_likely_team_sequence(log_probabilities[live],
                                                pipeline_config.model_switch_cost)
            labels[live] = np.where(states == _HOME_STATE, domain.LABEL_HOME, domain.LABEL_AWAY)
    return labels


def _most_likely_team_sequence(log_probabilities: NDArray[np.float64],
                               switch_cost: float) -> NDArray[np.int8]:
    """Two-state Viterbi. The loop runs over frames (~80k per half in total); each step is a
    handful of scalar operations, so a half decodes in well under a second."""
    n_frames = len(log_probabilities)
    came_from_other = np.zeros((n_frames, 2), dtype=bool)
    best_home, best_away = float(log_probabilities[0, 0]), float(log_probabilities[0, 1])
    home_scores, away_scores = log_probabilities[:, 0].tolist(), log_probabilities[:, 1].tolist()
    for frame in range(1, n_frames):
        home_from_away = best_away - switch_cost
        away_from_home = best_home - switch_cost
        came_from_other[frame, 0] = home_from_away > best_home
        came_from_other[frame, 1] = away_from_home > best_away
        best_home, best_away = (max(best_home, home_from_away) + home_scores[frame],
                                max(best_away, away_from_home) + away_scores[frame])

    states = np.zeros(n_frames, dtype=np.int8)
    states[-1] = _HOME_STATE if best_home >= best_away else _AWAY_STATE
    for frame in range(n_frames - 1, 0, -1):
        current = states[frame]
        states[frame - 1] = 1 - current if came_from_other[frame, current] else current
    return states
