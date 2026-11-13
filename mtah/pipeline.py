"""The processing steps from tracking + tagging to labels, in order. No I/O.

Entry points: compute_model_inputs, label_possessions.

Shared by run.py (the submission) and run_validation.py (the external validation games), so the
validated pipeline and the submitted pipeline cannot drift apart. The possession model is passed
in already trained: run.py trains it on all sample games, run_validation.py on all but the game
being scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier

from . import (config, dead_ball, decode_viterbi, domain, evidence, features, model_features,
               possession_model)


def compute_model_inputs(tracking: domain.TrackingData, frame_features: domain.FrameFeatures,
                         graded_halves: list[domain.GradedHalf],
                         pipeline_config: config.PipelineConfig) -> NDArray[np.float32]:
    """The model's feature matrix for one match. Also used to build training examples, so
    training and prediction see features computed the same way."""
    return model_features.compute_model_features(
        tracking, frame_features,
        evidence.detect_control_evidence(frame_features, pipeline_config),
        evidence.detect_touch_evidence(frame_features, pipeline_config), graded_halves)


def label_possessions(tracking: domain.TrackingData, tagging: pd.DataFrame,
                      graded_halves: list[domain.GradedHalf],
                      pipeline_config: config.PipelineConfig,
                      trained_model: HistGradientBoostingClassifier) -> domain.LabelPerFrame:
    frame_features = features.compute_frame_features(tracking, pipeline_config)
    is_dead = dead_ball.detect_dead_ball(frame_features, tagging, graded_halves, pipeline_config)
    home_probability = possession_model.predict_home_probability(
        trained_model, compute_model_inputs(tracking, frame_features, graded_halves,
                                            pipeline_config))
    return decode_viterbi.decode_possession_labels(home_probability, is_dead, graded_halves,
                                                   pipeline_config)
