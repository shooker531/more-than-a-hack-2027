"""Score the pipeline and the official baseline on Metrica's public sample games.

    .venv/bin/python run_validation.py

This is the number that decides whether an iteration is kept (ITERATIONS.md). It runs the exact
labelling code the submission uses (mtah.pipeline.label_possessions) on three other matches whose
tracking has been degraded to look like the challenge's, against possession labels derived from
their human-annotated events. Leave-one-game-out: each game is scored with a possession model
trained only on the other two, so no game is ever scored by a model that saw it. Results also go to evaluation/reports/validation_latest.json, which
scripts/log_experiment.py records.

main() reads top to bottom: load every game and build its training examples -> for each game:
train on the others -> derive truth -> for each seed: degrade -> synthesise tags -> label
(pipeline and baseline) -> then score everything and report.

Prerequisite (once): git clone https://github.com/metrica-sports/sample-data
data/external/metrica-sample-data
"""

from __future__ import annotations

import json
import os

import numpy as np

import games
from mtah import config, pipeline, possession_model
from validation import (degrade, labelled_games, reference_baseline, scoring, synthetic_tagging,
                        training_data)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SAMPLE_DATA_DIR = os.path.join(games.DATA_DIR, "external", "metrica-sample-data")
_DFL_DATA_DIR = os.path.join(games.DATA_DIR, "external", "idsse")
_CACHE_DIR = os.path.join(games.DATA_DIR, "cache", "validation")
_RESULTS_PATH = os.path.join(_REPO_ROOT, "evaluation", "reports", "validation_latest.json")
# Degradation is random (player dropouts, noise, tag jitter); several seeds per game average
# that out. The seeds are fixed so the validation score itself is reproducible.
_DEGRADATION_SEEDS = (0, 1, 2)
_BOOTSTRAP_DRAWS = 1000


def main(pipeline_config: config.PipelineConfig) -> None:
    validation_games = labelled_games.load_all_labelled_games(_SAMPLE_DATA_DIR, _DFL_DATA_DIR,
                                                              _CACHE_DIR)
    examples_by_game = {game.name: training_data.build_training_examples(game, pipeline_config)
                        for game in validation_games}

    truth_runs, pipeline_runs, baseline_runs, run_names = [], [], [], []
    for game in validation_games:
        held_out_training = training_data.combine_training_examples(
            [examples for other_game, examples in examples_by_game.items()
             if other_game != game.name])
        trained_model = possession_model.train_possession_model(
            held_out_training.feature_rows, held_out_training.is_home_in_possession,
            pipeline_config)
        for seed in _DEGRADATION_SEEDS:
            degraded_tracking = degrade.degrade_labelled_game(game, degrade.DEFAULT_DEGRADE_CONFIG,
                                                              seed)
            tagging = synthetic_tagging.synthesize_restart_tags(game.truth, game.graded_halves,
                                                                seed)
            pipeline_runs.append(pipeline.label_possessions(
                degraded_tracking, tagging, game.graded_halves, pipeline_config, trained_model))
            baseline_runs.append(reference_baseline.label_with_official_baseline(degraded_tracking))
            truth_runs.append(game.truth)
            run_names.append("{}/seed{}".format(game.name, seed))

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
    pooled_by_source = {
        source: scoring.score_labels(
            scoring.concatenate_label_runs([truth for name, truth in zip(run_names, truth_runs)
                                            if name.startswith(source)]),
            scoring.concatenate_label_runs([labels for name, labels
                                            in zip(run_names, pipeline_runs)
                                            if name.startswith(source)]), 0)
        for source in ("metrica", "dfl")}
    _write_results([game.name for game in validation_games], per_run_scores, pooled_scores,
                   pooled_by_source)
    _print_report(per_run_scores, pooled_scores, pooled_by_source)


def _write_results(game_names: list[str], per_run_scores: list[dict[str, object]],
                   pooled_scores: dict[str, dict[str, float]],
                   pooled_by_source: dict[str, dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(_RESULTS_PATH), exist_ok=True)
    with open(_RESULTS_PATH, "w") as handle:
        json.dump({"games": game_names, "seeds": list(_DEGRADATION_SEEDS),
                   "pooled": pooled_scores, "pooled_by_source": pooled_by_source,
                   "per_run": per_run_scores}, handle, indent=2)


def _print_report(per_run_scores: list[dict[str, object]],
                  pooled_scores: dict[str, dict[str, float]],
                  pooled_by_source: dict[str, dict[str, float]]) -> None:
    print("{:<18} {:>10} {:>10} {:>10} {:>10} {:>12}".format(
        "run", "spellF1", "base", "macroF1", "base", "poss pred/gt"))
    for row in per_run_scores:
        ours, theirs = row["pipeline"], row["baseline"]
        assert isinstance(ours, dict) and isinstance(theirs, dict)
        print("{:<18} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>7.0f}/{:<5.0f}".format(
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
    for source, scores in pooled_by_source.items():
        print("  {:<8} spellF1 {:.3f}  possessions {:.0f}/{:.0f}".format(
            source, scores["spell_f1"], scores["predicted_possessions"],
            scores["true_possessions"]))
    print("\nwrote", os.path.relpath(_RESULTS_PATH, _REPO_ROOT))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main(config.DEFAULT_CONFIG)
