"""The pipeline's output boundary: write the submission, validate it, summarise it.

Entry points: write_submission_csv, run_official_validator, summarise_possessions.

The summary prints aggregates only (counts, shares, durations), never per-frame labels, so it is
safe to share in an LLM conversation (see the mtah-data skill).
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from . import config, domain, runs

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VALIDATOR_SCRIPT = os.path.join(_REPO_ROOT, "evaluation", "validate_submission.py")
_LABEL_CODES = {domain.LABEL_HOME: 0, domain.LABEL_AWAY: 1, domain.LABEL_DEAD: 2,
                domain.LABEL_ABSTAIN: 3}


def write_submission_csv(output_path: str, labels: domain.LabelPerFrame) -> None:
    """Written by hand rather than with pandas so the bytes depend only on the labels (fixed
    '\\n' line endings, no index, no float formatting) — the file must regenerate identically."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(video_frame, label)
                          for video_frame, label in enumerate(labels))


def run_official_validator(submission_path: str) -> tuple[bool, str]:
    """Runs the organisers' script itself rather than reimplementing its checks, so a pass here
    means a pass at scoring time."""
    result = subprocess.run([sys.executable, _VALIDATOR_SCRIPT, submission_path],
                            capture_output=True, text=True)
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def summarise_possessions(labels: domain.LabelPerFrame,
                          graded_halves: list[domain.GradedHalf]) -> dict[str, float]:
    """Over graded frames only. Runs are computed per half so a possession never spans halftime."""
    possession_durations_s: list[float] = []
    dead_stretch_count = 0
    for graded_half in graded_halves:
        half_labels = labels[graded_half.first_video_frame:graded_half.last_video_frame + 1]
        half_codes = np.array([_LABEL_CODES[label] for label in half_labels], dtype=np.int8)
        for run_start, run_end, code in runs.equal_value_runs(half_codes):
            if code in (_LABEL_CODES[domain.LABEL_HOME], _LABEL_CODES[domain.LABEL_AWAY]):
                possession_durations_s.append((run_end - run_start) / config.FRAMES_PER_SECOND)
            dead_stretch_count += int(code == _LABEL_CODES[domain.LABEL_DEAD])

    graded_labels = np.concatenate([
        labels[graded_half.first_video_frame:graded_half.last_video_frame + 1]
        for graded_half in graded_halves])
    durations_s = np.array(possession_durations_s)
    return {
        "possessions": float(len(durations_s)),
        "possession_mean_s": float(durations_s.mean()),
        "possession_median_s": float(np.median(durations_s)),
        "possessions_under_2s": float(np.sum(durations_s < 2.0)),
        "dead_stretches": float(dead_stretch_count),
        "share_home": float(np.mean(graded_labels == domain.LABEL_HOME)),
        "share_away": float(np.mean(graded_labels == domain.LABEL_AWAY)),
        "share_dead": float(np.mean(graded_labels == domain.LABEL_DEAD)),
        "share_abstain": float(np.mean(graded_labels == domain.LABEL_ABSTAIN)),
    }
