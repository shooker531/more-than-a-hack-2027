"""Matches with known possession, in one common form, for training and validating the model.

Entry points: LabelledGame, DFL_MATCH_IDS, load_all_labelled_games, labelled_game_from_metrica,
load_dfl_game.

Two public sources, both other matches (allowed by RULES.md):
  - Metrica sample games 1-3: truth derived from their human-annotated events
    (validation/event_labels.py). The ball is already missing where an optical tracker loses it.
  - DFL / IDSSE open data, 7 Bundesliga matches (Bassek et al. 2025, CC-BY 4.0, data owner
    Deutsche Fussball Liga): TRACAB tracking at 25 Hz whose every ball frame carries
    BallPossession (team) and BallStatus (alive / dead) — the same kind of "corrected tracking"
    ownership the challenge's ground truth was built from. The ball is tracked on every frame,
    so degradation must remove it the way the challenge's tracker does.

DFL's BallPossession flips on almost every touch (37% of spells under 2 s, median 3.0 s, vs 12%
and 7.6 s in the challenge's public window), so the same controlled-possession rule used for the
Metrica truth is applied: a new team's spell only counts if it lasts >= 2 s of live play,
otherwise it merges back into the previous team's possession.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np
from kloppy import sportec

from mtah import domain, runs
from validation import event_labels, sample_games

DFL_MATCH_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
_DFL_COMPETITION_BY_MATCH = {"J03WMX": "000001", "J03WN1": "000001"}  # others: 2. Bundesliga
_DFL_SECOND_DIVISION = "000002"
_PITCH_LENGTH_M, _PITCH_WIDTH_M = 105.0, 68.0
_MIN_CONTROLLED_SPELL_FRAMES = 2 * 25


@dataclass(frozen=True)
class LabelledGame:
    """tracking is clean (before degradation); its speed channel is NaN — speeds are derived
    after degradation. is_ball_always_tracked marks sources whose ball must be removed during
    degradation to look like the challenge's automatic tracker."""
    name: str
    tracking: domain.TrackingData
    truth: domain.LabelPerFrame
    graded_halves: list[domain.GradedHalf]
    is_ball_always_tracked: bool


def load_all_labelled_games(sample_data_dir: str, dfl_data_dir: str,
                            cache_dir: str) -> list[LabelledGame]:
    """The fixed set used for training and validation: Metrica 1-3, then the 7 DFL matches.
    All must be present; a missing download raises rather than silently training on fewer."""
    metrica = [labelled_game_from_metrica(sample_games.load_sample_game(number, sample_data_dir,
                                                                        cache_dir))
               for number in sample_games.SAMPLE_GAME_NUMBERS]
    return metrica + [load_dfl_game(match_id, dfl_data_dir, cache_dir)
                      for match_id in DFL_MATCH_IDS]


def labelled_game_from_metrica(sample_game: sample_games.SampleGame) -> LabelledGame:
    truth = event_labels.derive_possession_labels(sample_game.events, sample_game.graded_halves,
                                                  len(sample_game.tracking.ball_norm))
    return LabelledGame(name="metrica{}".format(sample_game.game_number),
                        tracking=sample_game.tracking, truth=truth,
                        graded_halves=sample_game.graded_halves, is_ball_always_tracked=False)


def load_dfl_game(match_id: str, dfl_data_dir: str, cache_dir: str) -> LabelledGame:
    """kloppy takes ~2 minutes to parse one 400 MB positions file, so the result is cached."""
    cache_path = os.path.join(cache_dir, "dfl_{}.npz".format(match_id))
    if not os.path.isfile(cache_path):
        _parse_and_cache_dfl_game(match_id, dfl_data_dir, cache_path)
    with np.load(cache_path) as cached:
        tracking = domain.TrackingData(ball_norm=cached["ball"], players_norm=cached["players"],
                                       slot_team=cached["slot_team"],
                                       pitch_length_m=_PITCH_LENGTH_M,
                                       pitch_width_m=_PITCH_WIDTH_M)
        graded_halves = [domain.GradedHalf(int(half), int(first), int(last))
                         for half, first, last in cached["graded_halves"]]
        truth = cached["truth"].astype("<U1")
    return LabelledGame(name="dfl_{}".format(match_id), tracking=tracking, truth=truth,
                        graded_halves=graded_halves, is_ball_always_tracked=True)


def _parse_and_cache_dfl_game(match_id: str, dfl_data_dir: str, cache_path: str) -> None:
    competition = _DFL_COMPETITION_BY_MATCH.get(match_id, _DFL_SECOND_DIVISION)
    stem = "DFL-COM-{}_DFL-MAT-{}.xml".format(competition, match_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = sportec.load_tracking(
            meta_data=os.path.join(dfl_data_dir, "DFL_02_01_matchinformation_" + stem),
            raw_data=os.path.join(dfl_data_dir, "DFL_04_03_positions_raw_observed_" + stem),
            only_alive=False, coordinates="metrica")
        frame_table = dataset.to_df().reset_index(drop=True)

    slot_columns, slot_team = [], []
    for team_index, team in enumerate(dataset.metadata.teams):
        for player in team.players:
            if "{}_x".format(player.player_id) in frame_table.columns:
                slot_columns.append(str(player.player_id))
                slot_team.append(domain.HOME_TEAM if team_index == 0 else domain.AWAY_TEAM)
    n_frames = len(frame_table)
    players = np.full((n_frames, len(slot_columns), 3), np.nan, dtype=np.float32)
    for slot, player_id in enumerate(slot_columns):
        players[:, slot, 0] = frame_table[player_id + "_x"].to_numpy(dtype=np.float32)
        players[:, slot, 1] = frame_table[player_id + "_y"].to_numpy(dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    ball[:, 0] = frame_table["ball_x"].to_numpy(dtype=np.float32)
    ball[:, 1] = frame_table["ball_y"].to_numpy(dtype=np.float32)

    graded_halves = np.array([(int(period), int(rows.index.min()), int(rows.index.max()))
                              for period, rows in frame_table.groupby("period_id")],
                             dtype=np.int64)
    home_team_id = str(dataset.metadata.teams[0].team_id)
    is_alive = frame_table["ball_state"].astype(str).str.contains("alive").to_numpy()
    is_home_owner = (frame_table["ball_owning_team_id"].astype(str) == home_team_id).to_numpy()
    raw_team = np.where(is_home_owner, domain.HOME_TEAM, domain.AWAY_TEAM).astype(np.int8)
    truth = np.full(n_frames, domain.LABEL_ABSTAIN, dtype="<U1")
    for _, first, last in graded_halves:
        half = slice(first, last + 1)
        controlled_team = _merge_uncontrolled_spells(raw_team[half], is_alive[half])
        truth[half] = np.where(~is_alive[half], domain.LABEL_DEAD,
                               np.where(controlled_team == domain.HOME_TEAM, domain.LABEL_HOME,
                                        domain.LABEL_AWAY))

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, ball=ball, players=players,
             slot_team=np.array(slot_team, dtype=np.int8), graded_halves=graded_halves,
             truth=truth)


def _merge_uncontrolled_spells(raw_team: domain.TeamCodePerFrame,
                               is_alive: domain.BoolPerFrame) -> domain.TeamCodePerFrame:
    """A spell of a new team with fewer than 2 s of live play is a touch, not possession: it
    takes the previous holder's team. Length counts live frames only, so a ball touched just
    before it goes out does not count as kept."""
    merged = raw_team.copy()
    held_team = int(raw_team[0])
    for spell_start, spell_end, team in runs.equal_value_runs(raw_team):
        live_frames = int(np.sum(is_alive[spell_start:spell_end]))
        held_team = team if live_frames >= _MIN_CONTROLLED_SPELL_FRAMES else held_team
        merged[spell_start:spell_end] = held_team
    return merged
