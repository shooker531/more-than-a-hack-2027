"""Load the tactical ATD and Smart Tagging, indexed by video frame. The pipeline's input boundary.

Entry points: load_tracking, load_tagging, load_graded_halves.

Inputs are validated here, once, so later steps can assume well-formed data: the ATD must be one
record per video frame numbered from 1, and every Smart Tagging team value must be a known club.
"""

from __future__ import annotations

import io
import os
import sys
import xml.etree.ElementTree as ET
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import domain

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_REPO_ROOT, os.path.join(_REPO_ROOT, "starter")]
import games  # noqa: E402
import load_smart_tagging  # noqa: E402

# Smart Tagging club names, lower-cased. "N/A" (every throw-in, among others) means no team.
_TEAM_CODE_BY_TAG_VALUE = {"barcelona": domain.HOME_TEAM, "atletic club": domain.AWAY_TEAM,
                           "n/a": domain.NO_TEAM}


def load_graded_halves(game: str) -> list[domain.GradedHalf]:
    return [domain.GradedHalf(*block) for block in games.half_video_blocks(game)]


def load_tracking(game: str, data_dir: str, cache_dir: str) -> domain.TrackingData:
    """Parsing 199k lines takes a while, so the arrays are cached as .npz. The cache file name
    includes the raw file's size and mtime, so a changed export is re-parsed rather than served
    stale."""
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, data_dir)
    slot_team = _read_slot_teams(meta_path)
    raw_stat = os.stat(raw_path)
    cache_path = os.path.join(
        cache_dir, "{}_atd_{}_{}.npz".format(game, raw_stat.st_size, int(raw_stat.st_mtime)))
    if not os.path.isfile(cache_path):
        _parse_and_cache_raw_atd(raw_path, len(slot_team), games.source_frames("tactical", game),
                                 cache_path)
    with np.load(cache_path) as cached:
        ball_norm, players_norm = cached["ball"], cached["players"]
    pitch_length_m, pitch_width_m = cast("tuple[float, float]", games.GAMES[game]["pitch"])
    return domain.TrackingData(ball_norm=ball_norm, players_norm=players_norm,
                               slot_team=slot_team, pitch_length_m=pitch_length_m,
                               pitch_width_m=pitch_width_m)


def load_tagging(game: str, data_dir: str) -> pd.DataFrame:
    """Columns: code, set_piece_type, team (HOME_TEAM / AWAY_TEAM / NO_TEAM), start_video_frame,
    end_video_frame (exclusive). Codes and types are lower-cased because the export mixes upper
    and title case; matching is then case-insensitive by construction."""
    rows = []
    for instance in load_smart_tagging.load(game, data_dir):
        tags = {(group or "").strip().lower(): (value or "").strip().lower()
                for group, value in instance["tags"].items()}
        team_tag_value = tags.get("team", "n/a")
        if team_tag_value not in _TEAM_CODE_BY_TAG_VALUE:
            raise ValueError("unknown Smart Tagging team {!r}".format(team_tag_value))
        rows.append({
            "code": (instance["code"] or "").strip().lower(),
            "set_piece_type": tags.get("type", ""),
            "team": _TEAM_CODE_BY_TAG_VALUE[team_tag_value],
            "start_video_frame": instance["start_frame"],
            "end_video_frame": instance["end_frame"],
        })
    return (pd.DataFrame(rows).sort_values("start_video_frame", kind="stable")
            .reset_index(drop=True))


def _read_slot_teams(meta_path: str) -> NDArray[np.int8]:
    """The first <Team> in FIFA/EPTS metadata is the home side (kloppy follows the same rule)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(team_ids)))
    return np.array([domain.HOME_TEAM if player.get("teamId") == team_ids[0] else domain.AWAY_TEAM
                     for player in root.iter("Player")], dtype=np.int8)


def _parse_and_cache_raw_atd(raw_path: str, n_slots: int, n_video_frames: int,
                             cache_path: str) -> None:
    """Raw lines are "<frame_id>:<x,y,s>;...;<x,y,s>:<x,y,s>". Turning ':' and ';' into ',' makes
    each line a flat CSV row, so pandas' C parser reads the whole file in one pass instead of a
    Python loop over 199k lines."""
    with open(raw_path) as handle:
        flat_csv_text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(flat_csv_text), header=None, dtype=np.float32,
                        na_values=["NaN"], engine="c").to_numpy()
    expected_columns = 1 + 3 * (n_slots + 1)
    if table.shape[1] != expected_columns:
        raise ValueError("raw ATD has {} columns, expected {}".format(table.shape[1],
                                                                     expected_columns))
    atd_frame_ids = table[:, 0].astype(np.int64)
    if not np.array_equal(atd_frame_ids, np.arange(1, n_video_frames + 1)):
        raise ValueError("ATD is not one record per video frame numbered 1..{}".format(
            n_video_frames))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path,
             players=table[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3),
             ball=table[:, 1 + 3 * n_slots:])
