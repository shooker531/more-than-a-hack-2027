"""Smart-Tagging-like set-piece tags for the sample games, synthesised from their events.

Entry point: synthesize_set_piece_tags.

The pipeline reads only SET PIECES tags, so only those are produced, in the same columns as
mtah.load.load_tagging. They are made to share the real tags' imperfections, so validation does
not reward leaning on tags harder than the real ones allow:
  - windows open ~3.5 s before the restart (measured on the challenge kick-offs), with jitter;
  - throw-ins carry no team, as in the challenge export;
  - a share of non-kick-off set pieces is dropped (the real tags are automatic, with misses).
They have no false positives, so they remain somewhat cleaner than the real thing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mtah import config, domain

# Tag windows open ~3.5 s before the restart (starter/01_smart_tagging.ipynb), +/- 1.5 s jitter,
# and extend a little past it.
_OPENING_OFFSET_FRAMES = round(3.5 * config.FRAMES_PER_SECOND)
_OPENING_JITTER_FRAMES = round(1.5 * config.FRAMES_PER_SECOND)
_TAG_LENGTH_AFTER_RESTART_FRAMES = round(1.5 * config.FRAMES_PER_SECOND)
_MISSED_SET_PIECE_SHARE = 0.10


def synthesize_set_piece_tags(events: pd.DataFrame, graded_halves: list[domain.GradedHalf],
                              seed: int) -> pd.DataFrame:
    random_generator = np.random.default_rng(seed)
    is_set_piece = (events["type"] == "set piece") | (
        (events["type"] == "pass") & events["subtype"].str.contains("goal kick"))
    set_pieces = events[is_set_piece].reset_index(drop=True)
    set_piece_types = [_challenge_set_piece_type(subtype) for subtype in set_pieces["subtype"]]
    set_piece_types = _mark_first_kick_off_of_each_period(set_piece_types,
                                                          set_pieces["period"].to_numpy())

    opening_jitter = random_generator.integers(-_OPENING_JITTER_FRAMES,
                                               _OPENING_JITTER_FRAMES + 1, len(set_pieces))
    is_missed = random_generator.random(len(set_pieces)) < _MISSED_SET_PIECE_SHARE
    period_first_frame = {half.half: half.first_video_frame for half in graded_halves}
    restart_frames = set_pieces["start_video_frame"].to_numpy()
    tag_starts = np.maximum(
        restart_frames - _OPENING_OFFSET_FRAMES + opening_jitter,
        set_pieces["period"].map(period_first_frame).to_numpy())

    tags = pd.DataFrame({
        "code": "set pieces",
        "set_piece_type": set_piece_types,
        "team": np.where(np.array(set_piece_types) == "throw-in", domain.NO_TEAM,
                         set_pieces["team"].to_numpy()).astype(np.int8),
        "start_video_frame": tag_starts,
        "end_video_frame": restart_frames + _TAG_LENGTH_AFTER_RESTART_FRAMES,
    })
    is_kick_off = tags["set_piece_type"].str.startswith("kick off")
    return (tags[~is_missed | is_kick_off].sort_values("start_video_frame", kind="stable")
            .reset_index(drop=True))


def _challenge_set_piece_type(metrica_subtype: str) -> str:
    """Metrica subtype -> the challenge's lower-cased Smart Tagging type vocabulary."""
    vocabulary = (("kick off", "kick off (after goal)"), ("throw in", "throw-in"),
                  ("free kick", "free kick"), ("corner", "corner kick"),
                  ("goal kick", "goal kick"), ("penalty", "free kick"))
    matches = [challenge_type for metrica_word, challenge_type in vocabulary
               if metrica_word in metrica_subtype]
    if not matches:
        raise ValueError("unmapped set piece subtype {!r}".format(metrica_subtype))
    return matches[0]


def _mark_first_kick_off_of_each_period(set_piece_types: list[str],
                                        periods: NDArray[np.int64]) -> list[str]:
    """Metrica logs every kick-off alike; the challenge separates the half's opening kick-off
    ('kick off (start)', which the dead-ball detector relies on) from those after goals."""
    kick_off_indices = [index for index, set_piece_type in enumerate(set_piece_types)
                        if set_piece_type.startswith("kick off")]
    first_kick_off_per_period = {int(periods[index]): index for index in reversed(kick_off_indices)}
    return ["kick off (start)" if index in first_kick_off_per_period.values() else set_piece_type
            for index, set_piece_type in enumerate(set_piece_types)]
