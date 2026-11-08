"""The processing steps from tracking + tagging to labels, in order. No I/O.

Entry point: label_possessions.

Shared by run.py (the submission) and run_validation.py (the external validation games), so the
validated pipeline and the submitted pipeline cannot drift apart.
"""

from __future__ import annotations

import pandas as pd

from . import config, dead_ball, decode_rules, domain, evidence, features


def label_possessions(tracking: domain.TrackingData, tagging: pd.DataFrame,
                      graded_halves: list[domain.GradedHalf],
                      pipeline_config: config.PipelineConfig) -> domain.LabelPerFrame:
    frame_features = features.compute_frame_features(tracking, pipeline_config)
    is_dead = dead_ball.detect_dead_ball(frame_features, tagging, graded_halves, pipeline_config)
    control_evidence = evidence.detect_control_evidence(frame_features, pipeline_config)
    return decode_rules.decode_possession_labels(control_evidence, is_dead, tagging,
                                                 graded_halves, pipeline_config)
