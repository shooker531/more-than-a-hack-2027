"""Training examples for the possession model, built from the Metrica sample games.

Entry points: TRAINING_DEGRADATION_SEED, TrainingExamples, build_training_examples,
combine_training_examples.

The tracking is degraded exactly as in validation, so the model learns from the same imperfect
signals it will see on the challenge's automatic tracking. Labels are the event-derived
controlled-possession truth, live frames only (dead ball is decided by the rule detector).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from mtah import config, domain, features, pipeline
from validation import degrade, event_labels, sample_games

# Training always uses one fixed degradation; validation scores seeds 0-2 of games the model did
# not train on, so sharing seed 0 never leaks a scored game into training.
TRAINING_DEGRADATION_SEED = 0


class TrainingExamples(NamedTuple):
    feature_rows: NDArray[np.float32]
    is_home_in_possession: NDArray[np.bool_]


def build_training_examples(sample_game: sample_games.SampleGame,
                            pipeline_config: config.PipelineConfig) -> TrainingExamples:
    truth = event_labels.derive_possession_labels(sample_game.events, sample_game.graded_halves,
                                                  len(sample_game.tracking.ball_norm))
    degraded_tracking = degrade.degrade_tracking(sample_game.tracking,
                                                 degrade.DEFAULT_DEGRADE_CONFIG,
                                                 TRAINING_DEGRADATION_SEED)
    feature_matrix = pipeline.compute_model_inputs(
        degraded_tracking, features.compute_frame_features(degraded_tracking, pipeline_config),
        sample_game.graded_halves, pipeline_config)
    is_live = (truth == domain.LABEL_HOME) | (truth == domain.LABEL_AWAY)
    sampled_frames = np.flatnonzero(is_live)[::pipeline_config.model_training_frame_step]
    return TrainingExamples(feature_matrix[sampled_frames],
                            truth[sampled_frames] == domain.LABEL_HOME)


def combine_training_examples(examples: list[TrainingExamples]) -> TrainingExamples:
    return TrainingExamples(np.vstack([example.feature_rows for example in examples]),
                            np.concatenate([example.is_home_in_possession
                                            for example in examples]))
