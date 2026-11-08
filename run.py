"""Data bundle -> possession submission CSV. Deterministic: the same data gives the same bytes.

    .venv/bin/python run.py                              # evaluation/results/mtah_mvp.csv
    .venv/bin/python run.py --output path/to/team.csv

main() reads top to bottom: load inputs -> label (mtah/pipeline.py lists the processing steps)
-> write -> validate -> report.
"""

from __future__ import annotations

import argparse
import os
import sys

import games
from mtah import config, load, pipeline, write

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUTPUT_PATH = os.path.join(_REPO_ROOT, "evaluation", "results", "mtah_mvp.csv")
_CACHE_DIR = os.path.join(games.DATA_DIR, "cache")


def main(output_path: str, pipeline_config: config.PipelineConfig) -> None:
    tracking = load.load_tracking(games.GAME, games.DATA_DIR, _CACHE_DIR)
    tagging = load.load_tagging(games.GAME, games.DATA_DIR)
    graded_halves = load.load_graded_halves(games.GAME)

    labels = pipeline.label_possessions(tracking, tagging, graded_halves, pipeline_config)

    write.write_submission_csv(output_path, labels)
    is_valid, validator_report = write.run_official_validator(output_path)
    _print_report(output_path, is_valid, validator_report,
                  write.summarise_possessions(labels, graded_halves))


def _print_report(output_path: str, is_valid: bool, validator_report: str,
                  summary: dict[str, float]) -> None:
    print("wrote {} | valid: {}".format(os.path.relpath(output_path, _REPO_ROOT), is_valid))
    if not is_valid:
        print(validator_report)
        sys.exit(1)
    for name, value in summary.items():
        print("  {:<22} {:.3f}".format(name, value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=_DEFAULT_OUTPUT_PATH)
    main(parser.parse_args().output, config.DEFAULT_CONFIG)
