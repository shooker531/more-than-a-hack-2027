"""Run-length helpers. Every interval returned is (start, end) with end exclusive."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def true_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.diff(padded)
    return [(int(start), int(end))
            for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def drop_true_runs_shorter_than(mask: NDArray[np.bool_], min_length: int) -> NDArray[np.bool_]:
    cleaned = mask.copy()
    for start, end in true_runs(mask):
        cleaned[start:end] = end - start >= min_length
    return cleaned


def fill_false_runs_shorter_than(mask: NDArray[np.bool_], min_length: int) -> NDArray[np.bool_]:
    """Only interior gaps are filled: a False run touching either end of the array is not
    bounded by True on both sides, so there is nothing to bridge."""
    filled = mask.copy()
    for start, end in true_runs(~mask):
        is_interior = start > 0 and end < len(mask)
        filled[start:end] = is_interior and end - start < min_length
    return filled


def equal_value_runs(values: NDArray[np.int8]) -> list[tuple[int, int, int]]:
    """(start, end, value) for each maximal run of one value."""
    change_points = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [len(values)]))
    return [(int(start), int(end), int(values[start]))
            for start, end in zip(starts, ends) if end > start]
