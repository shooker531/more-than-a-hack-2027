"""The learned per-frame possession model: P(home has the ball) from model_features.

Entry points: train_possession_model, predict_home_probability.

Trained only on other matches (the Metrica sample games, labels from their human-annotated
events), never on the challenge match — RULES.md forbids labels on the provided game. Training
is deterministic (fixed seed, single model), so run.py regenerates the same submission.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config, domain


def train_possession_model(feature_rows: NDArray[np.float32],
                           is_home_in_possession: NDArray[np.bool_],
                           pipeline_config: config.PipelineConfig
                           ) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_iter=pipeline_config.model_max_iterations,
        learning_rate=pipeline_config.model_learning_rate,
        max_leaf_nodes=pipeline_config.model_max_leaf_nodes,
        l2_regularization=pipeline_config.model_l2_regularisation,
        random_state=pipeline_config.model_random_seed)
    model.fit(feature_rows, is_home_in_possession.astype(np.int8))
    return model


def predict_home_probability(model: HistGradientBoostingClassifier,
                             model_feature_matrix: NDArray[np.float32]) -> domain.FloatPerFrame:
    probabilities: domain.FloatPerFrame = model.predict_proba(model_feature_matrix)[:, 1]
    return probabilities
