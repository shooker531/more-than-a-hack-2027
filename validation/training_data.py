"""Training examples for the possession model, built from labelled public matches.

Entry points: TrainingExamples, build_training_examples, combine_training_examples.

The tracking is degraded exactly as in validation, so the model learns from the same imperfect
signals it will see on the challenge's automatic tracking. Labels are the controlled-possession
truth, live frames only (dead ball is decided by the rule detector).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from mtah import config, domain, features, pipeline
from validation import degrade, labelled_games

# Training copies use degradation seeds 0, 1, ...; validation scores seeds 0-2 of games the model
# did not train on, so sharing seed numbers never leaks a scored game into training.


class TrainingExamples(NamedTuple):
    feature_rows: NDArray[np.float32]
    is_home_in_possession: NDArray[np.bool_]


def build_training_examples(game: labelled_games.LabelledGame,
                            pipeline_config: config.PipelineConfig) -> TrainingExamples:
    is_live = (game.truth == domain.LABEL_HOME) | (game.truth == domain.LABEL_AWAY)
    sampled_frames = np.flatnonzero(is_live)[::pipeline_config.model_training_frame_step]
    return combine_training_examples([
        _examples_from_one_degraded_copy(game, sampled_frames, seed, pipeline_config)
        for seed in range(pipeline_config.model_training_degraded_copies)])


def combine_training_examples(examples: list[TrainingExamples]) -> TrainingExamples:
    return TrainingExamples(np.vstack([example.feature_rows for example in examples]),
                            np.concatenate([example.is_home_in_possession
                                            for example in examples]))


def _examples_from_one_degraded_copy(game: labelled_games.LabelledGame,
                                     sampled_frames: NDArray[np.int64], seed: int,
                                     pipeline_config: config.PipelineConfig) -> TrainingExamples:
    degraded_tracking = degrade.degrade_labelled_game(game, degrade.DEFAULT_DEGRADE_CONFIG, seed)
    feature_matrix = pipeline.compute_model_inputs(
        degraded_tracking, features.compute_frame_features(degraded_tracking, pipeline_config),
        game.graded_halves, pipeline_config)
    return TrainingExamples(feature_matrix[sampled_frames],
                            game.truth[sampled_frames] == domain.LABEL_HOME)
