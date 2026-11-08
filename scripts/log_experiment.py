"""Append one row to experiments.csv describing the current pipeline output.

    .venv/bin/python run.py
    .venv/bin/python scripts/log_experiment.py --iteration 001 --title "MVP state machine"

Numbers are computed from the written submission and the config, never typed by hand, so the
log in ITERATIONS.md can quote them without drifting from reality. Only aggregates are recorded:
experiments.csv is safe to commit and to share.

Validation columns stay empty until the external validation harness exists; from then on they
are the numbers that decide whether an iteration is kept.
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
_VALIDATION_COLUMNS = ["validation_games", "validation_spell_f1", "validation_macro_f1"]


def main(iteration: str, title: str, submission_path: str) -> None:
    labels = pd.read_csv(submission_path)["label"].to_numpy().astype("<U1")
    full_match_summary = write.summarise_possessions(labels, load.load_graded_halves(games.GAME))
    row = {
        "iteration": iteration,
        "title": title,
        "logged_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M"),
        "git_commit": _current_git_commit(),
        **{column: "" for column in _VALIDATION_COLUMNS},
        **{"full_match_" + name: round(value, 4) for name, value in full_match_summary.items()},
        "config_json": json.dumps(dataclasses.asdict(config.DEFAULT_CONFIG), sort_keys=True),
    }
    _append_row(_EXPERIMENTS_CSV, row)
    print("logged iteration {} to {}".format(iteration, os.path.relpath(_EXPERIMENTS_CSV,
                                                                        _REPO_ROOT)))


def _current_git_commit() -> str:
    """Suffixed with '+dirty' when there are uncommitted changes, so a row can never silently
    point at code that differs from what produced it."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                            cwd=_REPO_ROOT, capture_output=True, text=True, check=True).stdout
    return commit + ("+dirty" if status.strip() else "")


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
