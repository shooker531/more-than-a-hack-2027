"""Load Metrica's public sample games into the same structures the pipeline uses.

Entry points: SAMPLE_GAME_NUMBERS, load_sample_game.

Source: https://github.com/metrica-sports/sample-data, cloned into data/external/ (gitignored).
Games 1-2 are Metrica CSV tracking + CSV events; game 3 is FIFA/EPTS tracking (the challenge's
format) + JSON events. kloppy reads all three tracking files; events are normalised here.

Differences from the challenge ATD, handled downstream by validation/degrade.py: slots are real
players (not tracklets), there is no speed channel, and the ball is cleaner.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from kloppy import metrica

from mtah import domain

SAMPLE_GAME_NUMBERS = (1, 2, 3)

# Metrica sample games are played on a standard 105 x 68 m pitch (game 3 metadata FieldSize).
_PITCH_LENGTH_M, _PITCH_WIDTH_M = 105.0, 68.0
_EPTS_HOME_TEAM_ID = "FIFATMA"


@dataclass(frozen=True)
class SampleGame:
    """tracking.players_norm speed channel is NaN here: speeds are derived after degradation, from
    the same noisy positions the pipeline would see.

    events columns: team (HOME/AWAY/NO_TEAM), type, subtype (lower case, '' when none),
    start_video_frame, end_video_frame (0-based, end >= start), period.
    """
    game_number: int
    tracking: domain.TrackingData
    events: pd.DataFrame
    graded_halves: list[domain.GradedHalf]


def load_sample_game(game_number: int, sample_data_dir: str, cache_dir: str) -> SampleGame:
    """kloppy takes ~15 s per game, so the parsed arrays and events are cached."""
    cache_stem = os.path.join(cache_dir, "metrica_sample_game_{}".format(game_number))
    if not os.path.isfile(cache_stem + ".npz"):
        _parse_and_cache_sample_game(game_number, sample_data_dir, cache_stem)
    with np.load(cache_stem + ".npz") as cached:
        tracking = domain.TrackingData(
            ball_norm=cached["ball"], players_norm=cached["players"],
            slot_team=cached["slot_team"], pitch_length_m=_PITCH_LENGTH_M,
            pitch_width_m=_PITCH_WIDTH_M)
        graded_halves = [domain.GradedHalf(int(half), int(first), int(last))
                         for half, first, last in cached["graded_halves"]]
    events = pd.read_csv(cache_stem + "_events.csv", keep_default_na=False)
    return SampleGame(game_number=game_number, tracking=tracking, events=events,
                      graded_halves=graded_halves)


def _parse_and_cache_sample_game(game_number: int, sample_data_dir: str, cache_stem: str) -> None:
    game_dir = os.path.join(sample_data_dir, "data", "Sample_Game_{}".format(game_number))
    prefix = os.path.join(game_dir, "Sample_Game_{}".format(game_number))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if game_number == 3:
            dataset = metrica.load_tracking_epts(meta_data=prefix + "_metadata.xml",
                                                 raw_data=prefix + "_tracking.txt")
            events = _normalise_json_events(prefix + "_events.json")
        else:
            dataset = metrica.load_tracking_csv(home_data=prefix + "_RawTrackingData_Home_Team.csv",
                                                away_data=prefix + "_RawTrackingData_Away_Team.csv",
                                                coordinates="metrica")
            events = _normalise_csv_events(prefix + "_RawEventsData.csv")
        frame_table = dataset.to_df()

    expected_frame_ids = np.arange(1, len(frame_table) + 1)
    if not np.array_equal(frame_table["frame_id"].to_numpy(), expected_frame_ids):
        raise ValueError("sample game {} frames are not contiguous from 1".format(game_number))

    slot_columns, slot_team = [], []
    for team_index, team in enumerate(dataset.metadata.teams):
        for player in team.players:
            slot_columns.append(str(player.player_id))
            slot_team.append(domain.HOME_TEAM if team_index == 0 else domain.AWAY_TEAM)
    n_frames, n_slots = len(frame_table), len(slot_columns)
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    for slot, player_id in enumerate(slot_columns):
        players[:, slot, 0] = frame_table[player_id + "_x"].to_numpy(dtype=np.float32)
        players[:, slot, 1] = frame_table[player_id + "_y"].to_numpy(dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    ball[:, 0] = frame_table["ball_x"].to_numpy(dtype=np.float32)
    ball[:, 1] = frame_table["ball_y"].to_numpy(dtype=np.float32)

    period_frames = frame_table.groupby("period_id")["frame_id"].agg(["min", "max"])
    graded_halves = np.array([(int(period), int(row["min"]) - 1, int(row["max"]) - 1)
                              for period, row in period_frames.iterrows()], dtype=np.int64)

    os.makedirs(os.path.dirname(cache_stem), exist_ok=True)
    np.savez(cache_stem + ".npz", ball=ball, players=players,
             slot_team=np.array(slot_team, dtype=np.int8), graded_halves=graded_halves)
    events.to_csv(cache_stem + "_events.csv", index=False)


def _normalise_csv_events(events_path: str) -> pd.DataFrame:
    raw = pd.read_csv(events_path)
    team_code = {"home": domain.HOME_TEAM, "away": domain.AWAY_TEAM}
    return _finalise_events(pd.DataFrame({
        "team": raw["Team"].str.lower().map(team_code),
        "type": raw["Type"].str.lower(),
        "subtype": raw["Subtype"].fillna("").str.lower(),
        "start_frame_1based": raw["Start Frame"],
        "end_frame_1based": raw["End Frame"],
        "period": raw["Period"],
    }))


def _normalise_json_events(events_path: str) -> pd.DataFrame:
    with open(events_path) as handle:
        raw_events = json.load(handle)["data"]
    rows = []
    for event in raw_events:
        subtypes = event["subtypes"] or []
        subtype_names = [subtypes["name"]] if isinstance(subtypes, dict) else [
            subtype["name"] for subtype in subtypes]
        rows.append({
            "team": (domain.HOME_TEAM if event["team"]["id"] == _EPTS_HOME_TEAM_ID
                     else domain.AWAY_TEAM),
            "type": event["type"]["name"].lower(),
            "subtype": "-".join(subtype_names).lower(),
            "start_frame_1based": event["start"]["frame"],
            "end_frame_1based": event["end"]["frame"],
            "period": event["period"],
        })
    return _finalise_events(pd.DataFrame(rows))


def _finalise_events(events: pd.DataFrame) -> pd.DataFrame:
    """Kick-offs are logged with end frame 0, hence the max: an event never ends before it
    starts. Frames move to 0-based to match the tracking arrays."""
    start = events["start_frame_1based"].astype(np.int64) - 1
    end = np.maximum(events["end_frame_1based"].astype(np.int64) - 1, start)
    return (pd.DataFrame({
        "team": events["team"].astype(np.int8),
        "type": events["type"],
        "subtype": events["subtype"],
        "start_video_frame": start,
        "end_video_frame": end,
        "period": events["period"].astype(np.int64),
    }).sort_values("start_video_frame", kind="stable").reset_index(drop=True))
