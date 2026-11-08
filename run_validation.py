"""Score the pipeline and the official baseline on Metrica's public sample games.

    .venv/bin/python run_validation.py

This is the number that decides whether an iteration is kept (ITERATIONS.md). It runs the exact
labelling code the submission uses (mtah.pipeline.label_possessions) on three other matches whose
tracking has been degraded to look like the challenge's, against possession labels derived from
their human-annotated events. Results also go to evaluation/reports/validation_latest.json, which
scripts/log_experiment.py records.

main() reads top to bottom: for each game, load -> derive truth -> for each seed: degrade ->
synthesise tags -> label (pipeline and baseline) -> then score everything and report.

Prerequisite (once): git clone https://github.com/metrica-sports/sample-data
data/external/metrica-sample-data
"""

from __future__ import annotations

import json
import os

import numpy as np

import games
from mtah import config, pipeline
from validation import (degrade, event_labels, reference_baseline, sample_games, scoring,
                        synthetic_tagging)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SAMPLE_DATA_DIR = os.path.join(games.DATA_DIR, "external", "metrica-sample-data")
_CACHE_DIR = os.path.join(games.DATA_DIR, "cache", "validation")
_RESULTS_PATH = os.path.join(_REPO_ROOT, "evaluation", "reports", "validation_latest.json")
# Degradation is random (player dropouts, noise, tag jitter); several seeds per game average
# that out. The seeds are fixed so the validation score itself is reproducible.
_DEGRADATION_SEEDS = (0, 1, 2)
_BOOTSTRAP_DRAWS = 1000


def main(pipeline_config: config.PipelineConfig) -> None:
    truth_runs, pipeline_runs, baseline_runs, run_names = [], [], [], []
    for game_number in sample_games.SAMPLE_GAME_NUMBERS:
        sample_game = sample_games.load_sample_game(game_number, _SAMPLE_DATA_DIR, _CACHE_DIR)
        truth = event_labels.derive_possession_labels(
            sample_game.events, sample_game.graded_halves, len(sample_game.tracking.ball_norm))
        for seed in _DEGRADATION_SEEDS:
            degraded_tracking = degrade.degrade_tracking(
                sample_game.tracking, degrade.DEFAULT_DEGRADE_CONFIG, seed)
            tagging = synthetic_tagging.synthesize_set_piece_tags(
                sample_game.events, sample_game.graded_halves, seed)
            pipeline_runs.append(pipeline.label_possessions(
                degraded_tracking, tagging, sample_game.graded_halves, pipeline_config))
            baseline_runs.append(reference_baseline.label_with_official_baseline(degraded_tracking))
            truth_runs.append(truth)
            run_names.append("game{}/seed{}".format(game_number, seed))

    per_run_scores: list[dict[str, object]] = [
        {"run": name,
         "pipeline": scoring.score_labels(truth, predicted_by_pipeline, bootstrap_draws=0),
         "baseline": scoring.score_labels(truth, predicted_by_baseline, bootstrap_draws=0)}
        for name, truth, predicted_by_pipeline, predicted_by_baseline
        in zip(run_names, truth_runs, pipeline_runs, baseline_runs)]
    pooled_truth = scoring.concatenate_label_runs(truth_runs)
    pooled_scores = {
        "pipeline": scoring.score_labels(pooled_truth, scoring.concatenate_label_runs(
            pipeline_runs), _BOOTSTRAP_DRAWS),
        "baseline": scoring.score_labels(pooled_truth, scoring.concatenate_label_runs(
            baseline_runs), _BOOTSTRAP_DRAWS),
    }
    _write_results(per_run_scores, pooled_scores)
    _print_report(per_run_scores, pooled_scores)


def _write_results(per_run_scores: list[dict[str, object]],
                   pooled_scores: dict[str, dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(_RESULTS_PATH), exist_ok=True)
    with open(_RESULTS_PATH, "w") as handle:
        json.dump({"games": list(sample_games.SAMPLE_GAME_NUMBERS),
                   "seeds": list(_DEGRADATION_SEEDS),
                   "pooled": pooled_scores, "per_run": per_run_scores}, handle, indent=2)


def _print_report(per_run_scores: list[dict[str, object]],
                  pooled_scores: dict[str, dict[str, float]]) -> None:
    print("{:<14} {:>10} {:>10} {:>10} {:>10} {:>12}".format(
        "run", "spellF1", "base", "macroF1", "base", "poss pred/gt"))
    for row in per_run_scores:
        ours, theirs = row["pipeline"], row["baseline"]
        assert isinstance(ours, dict) and isinstance(theirs, dict)
        print("{:<14} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>7.0f}/{:<5.0f}".format(
            str(row["run"]), ours["spell_f1"], theirs["spell_f1"], ours["macro_f1"],
            theirs["macro_f1"], ours["predicted_possessions"], ours["true_possessions"]))
    print()
    for name, scores in pooled_scores.items():
        print("POOLED {:<9} spellF1 {:.3f} [{:.3f}, {:.3f}]  macroF1 {:.3f}  alive/dead F1 {:.3f}"
              "  team acc {:.3f}  possessions {:.0f}/{:.0f}".format(
                  name, scores["spell_f1"], scores["spell_f1_low"], scores["spell_f1_high"],
                  scores["macro_f1"], scores["alive_dead_f1"],
                  scores["team_accuracy_when_alive"], scores["predicted_possessions"],
                  scores["true_possessions"]))
    print("\nwrote", os.path.relpath(_RESULTS_PATH, _REPO_ROOT))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main(config.DEFAULT_CONFIG)
