"""Make sample-game tracking look like the challenge's automatic tracking. Seeded, so repeatable.

Entry points: DegradeConfig, DEFAULT_DEGRADE_CONFIG, degrade_tracking.

The ball is left as it is. Measured on the sample games, the ball is already missing on 32-41% of
frames (challenge: 41%), in ~160 gaps (challenge: 150), and the gaps line up with stoppages the
same way: players are slow on 52-56% of untracked-ball frames vs 4-11% of tracked ones (challenge:
47% vs 10%). Adding more ball dropout would make the validation games less like the challenge,
not more.

What does differ, and is degraded here: sample players are tracked cleanly every frame and there is
no speed channel. The challenge tracks ~20.5 of 22 players per frame and supplies speeds.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray

from mtah import config, domain


@dataclasses.dataclass(frozen=True)
class DegradeConfig:
    # Challenge ATD tracks a median of 11 of 11 players per team but p25 of 10 and p5 of 6-7, so
    # roughly 8% of player-frames are missing, in runs (players lost in crowds, at the edge of
    # the camera view). Run length is not measurable from the ATD; ~2 s is a plausible occlusion.
    player_missing_share: float = 0.08
    player_dropout_median_frames: float = 50.0
    player_dropout_log_sigma: float = 1.0
    # Automatic tracking jitters; clean sample positions do not. 0.1 m per axis is a mild,
    # deliberately conservative perturbation.
    position_noise_sd_m: float = 0.1
    # Speeds are derived from positions over this centred window, as a tracker would smooth them.
    speed_window_frames: int = 5


DEFAULT_DEGRADE_CONFIG = DegradeConfig()


def degrade_tracking(clean_tracking: domain.TrackingData, degrade_config: DegradeConfig,
                     seed: int) -> domain.TrackingData:
    random_generator = np.random.default_rng(seed)
    players_norm = clean_tracking.players_norm.astype(np.float64).copy()
    noise_sd_norm = np.array([degrade_config.position_noise_sd_m / clean_tracking.pitch_length_m,
                              degrade_config.position_noise_sd_m / clean_tracking.pitch_width_m])
    players_norm[:, :, :2] += (
        random_generator.normal(0.0, 1.0, players_norm[:, :, :2].shape) * noise_sd_norm)
    players_norm[_player_dropout_mask(players_norm.shape[0], players_norm.shape[1],
                                      degrade_config, random_generator)] = np.nan
    players_norm[:, :, 2] = _speed_from_positions_mps(players_norm, clean_tracking,
                                                      degrade_config.speed_window_frames)
    return dataclasses.replace(clean_tracking, players_norm=players_norm.astype(np.float32))


def _player_dropout_mask(n_frames: int, n_slots: int, degrade_config: DegradeConfig,
                         random_generator: np.random.Generator) -> NDArray[np.bool_]:
    """Independent dropout runs per slot until each slot has lost its target share of frames.
    Applied to every slot, including substitutes' (already NaN while off the pitch)."""
    is_dropped = np.zeros((n_frames, n_slots), dtype=bool)
    target_dropped_frames = int(degrade_config.player_missing_share * n_frames)
    for slot in range(n_slots):
        while is_dropped[:, slot].sum() < target_dropped_frames:
            run_length = int(random_generator.lognormal(
                np.log(degrade_config.player_dropout_median_frames),
                degrade_config.player_dropout_log_sigma)) + 1
            run_start = int(random_generator.integers(0, n_frames))
            is_dropped[run_start:run_start + run_length, slot] = True
    return is_dropped


def _speed_from_positions_mps(players_norm: NDArray[np.float64],
                              tracking: domain.TrackingData,
                              window_frames: int) -> NDArray[np.float64]:
    half_window = window_frames // 2
    span = 2 * half_window
    speed_mps = np.full(players_norm.shape[:2], np.nan)
    displacement_m = np.hypot(
        (players_norm[span:, :, 0] - players_norm[:-span, :, 0]) * tracking.pitch_length_m,
        (players_norm[span:, :, 1] - players_norm[:-span, :, 1]) * tracking.pitch_width_m)
    speed_mps[half_window:half_window + len(displacement_m)] = (
        displacement_m * config.FRAMES_PER_SECOND / span)
    return speed_mps
