"""Score labels with the organisers' own metric code.

Entry points: score_labels, concatenate_label_runs.

Pooling: several (game, seed) runs are joined into one array with an X frame between them. The
official scorer computes everything inside non-X runs, so nothing leaks across the join, and the
bootstrap then resamples possessions from all games at once — far more possessions than any single
game, so a tighter interval.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from mtah import domain

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "evaluation"))
import possession_metrics  # noqa: E402


def score_labels(truth_labels: domain.LabelPerFrame, predicted_labels: domain.LabelPerFrame,
                 bootstrap_draws: int) -> dict[str, float]:
    """bootstrap_draws = 0 skips the interval (low/high come back NaN)."""
    result = possession_metrics.evaluate(possession_metrics.label_codes(truth_labels),
                                         possession_metrics.label_codes(predicted_labels),
                                         bootstrap_n=bootstrap_draws)
    interval_low, interval_high = result["spell_bootstrap_low"], result["spell_bootstrap_high"]
    return {
        "spell_f1": float(result["spell_f1_mean"]),
        "spell_f1_low": float("nan") if interval_low is None else float(interval_low),
        "spell_f1_high": float("nan") if interval_high is None else float(interval_high),
        "macro_f1": float(result["macro_f1"]),
        "alive_dead_f1": float(result["alive_dead_f1"]),
        "team_accuracy_when_alive": float(result["team_acc_gt_alive"]),
        "true_possessions": float(result["n_gt_spells"]),
        "predicted_possessions": float(result["n_pred_spells"]),
    }


def concatenate_label_runs(label_runs: list[domain.LabelPerFrame]) -> domain.LabelPerFrame:
    separator = np.array([domain.LABEL_ABSTAIN], dtype="<U1")
    pieces: list[domain.LabelPerFrame] = []
    for label_run in label_runs:
        pieces += [label_run, separator]
    return np.concatenate(pieces)
