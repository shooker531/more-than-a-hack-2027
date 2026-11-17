"""Append one row to experiments.csv describing the current pipeline output.

    .venv/bin/python run.py
    .venv/bin/python scripts/log_experiment.py --iteration 001 --title "MVP state machine"

Numbers are computed from the written submission and the config, never typed by hand, so the
log in ITERATIONS.md can quote them without drifting from reality. Only aggregates are recorded:
experiments.csv is safe to commit and to share.

Validation columns come from evaluation/reports/validation_latest.json, written by
run_validation.py — run it first. They are the numbers that decide whether an iteration is kept.

Code version: kept iterations are committed first, so the row records a clean commit. A rejected
iteration is never committed: its uncommitted change is saved as experiments/patches/<id>.patch
(applies on top of the recorded base commit) and the row points at it, so the code is preserved
for the write-up without a commit-and-revert round trip.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import json
import os
import subprocess
import sys

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
import games  # noqa: E402
from mtah import config, load, write  # noqa: E402

_EXPERIMENTS_CSV = os.path.join(_REPO_ROOT, "experiments.csv")
_DEFAULT_SUBMISSION = os.path.join(_REPO_ROOT, "evaluation", "results", "mtah_mvp.csv")
_VALIDATION_RESULTS = os.path.join(_REPO_ROOT, "evaluation", "reports", "validation_latest.json")
_PATCH_DIR = os.path.join(_REPO_ROOT, "experiments", "patches")
# Log files are always edited after the run they describe, so they are never part of the code.
_LOG_PATHSPEC = [".", ":!experiments.csv", ":!ITERATIONS.md", ":!experiments/"]


def main(iteration: str, title: str, submission_path: str) -> None:
    labels = pd.read_csv(submission_path)["label"].to_numpy().astype("<U1")
    full_match_summary = write.summarise_possessions(labels, load.load_graded_halves(games.GAME))
    row = {
        "iteration": iteration,
        "title": title,
        "logged_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M"),
        "git_commit": _record_code_version(iteration),
        **_validation_columns(_VALIDATION_RESULTS),
        **{"full_match_" + name: round(value, 4) for name, value in full_match_summary.items()},
        "config_json": json.dumps(dataclasses.asdict(config.DEFAULT_CONFIG), sort_keys=True),
    }
    _append_row(_EXPERIMENTS_CSV, row)
    print("logged iteration {} to {}".format(iteration, os.path.relpath(_EXPERIMENTS_CSV,
                                                                        _REPO_ROOT)))


def _validation_columns(results_path: str) -> dict[str, object]:
    with open(results_path) as handle:
        results = json.load(handle)
    ours, official_baseline = results["pooled"]["pipeline"], results["pooled"]["baseline"]
    return {
        "validation_runs": "games {} x seeds {}".format(results["games"], results["seeds"]),
        "validation_spell_f1": round(ours["spell_f1"], 4),
        "validation_spell_f1_low": round(ours["spell_f1_low"], 4),
        "validation_spell_f1_high": round(ours["spell_f1_high"], 4),
        "validation_macro_f1": round(ours["macro_f1"], 4),
        "validation_alive_dead_f1": round(ours["alive_dead_f1"], 4),
        "validation_baseline_spell_f1": round(official_baseline["spell_f1"], 4),
    }


def _record_code_version(iteration: str) -> str:
    """'<commit>' when the code is committed, else '<base commit>+patch:<path>' after saving the
    uncommitted change (including new, untracked files) as a patch."""
    base_commit = _git(["rev-parse", "--short", "HEAD"]).strip()
    untracked_files = _git(["ls-files", "--others", "--exclude-standard", "--", *_LOG_PATHSPEC])
    tracked_diff = _git(["diff", "HEAD", "--", *_LOG_PATHSPEC])
    untracked_diff = "".join(_git(["diff", "--no-index", "--", os.devnull, path], check=False)
                             for path in untracked_files.split())
    patch_text = tracked_diff + untracked_diff
    patch_path = os.path.join(_PATCH_DIR, "{}.patch".format(iteration))
    if patch_text:
        os.makedirs(_PATCH_DIR, exist_ok=True)
        with open(patch_path, "w") as handle:
            handle.write(patch_text)
    return (base_commit + "+patch:" + os.path.relpath(patch_path, _REPO_ROOT) if patch_text
            else base_commit)


def _git(arguments: list[str], check: bool = True) -> str:
    """git diff --no-index exits 1 when files differ, which is its normal answer, hence check."""
    return subprocess.run(["git", *arguments], cwd=_REPO_ROOT, capture_output=True, text=True,
                          check=check).stdout


def _append_row(csv_path: str, row: dict[str, object]) -> None:
    """Raises if the columns changed, rather than writing a misaligned row."""
    is_new_file = not os.path.isfile(csv_path)
    if not is_new_file:
        with open(csv_path, newline="") as handle:
            existing_columns = next(csv.reader(handle))
        if existing_columns != list(row):
            raise ValueError("experiments.csv columns differ from this row; migrate the file first")
    with open(csv_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iteration", required=True, help="zero-padded id, e.g. 002")
    parser.add_argument("--title", required=True)
    parser.add_argument("--submission", default=_DEFAULT_SUBMISSION)
    arguments = parser.parse_args()
    main(arguments.iteration, arguments.title, arguments.submission)
