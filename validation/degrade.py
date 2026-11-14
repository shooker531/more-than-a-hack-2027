"""Make sample-game tracking look like the challenge's automatic tracking. Seeded, so repeatable.

Entry points: DegradeConfig, DEFAULT_DEGRADE_CONFIG, degrade_labelled_game, degrade_tracking.

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

from mtah import config, domain, runs
from validation import labelled_games


@dataclasses.dataclass(frozen=True)
class DegradeConfig:
    # Challenge ATD tracks a median of 11 of 11 players per team but p25 of 10 and p5 of 6-7, so
    # roughly 8% of player-frames are missing, in runs (players lost in crowds, at the edge of
    # the camera view). Run length is not measurable from the ATD; ~2 s is a plausible occlusion.
    player_missing_share: float = 0.08
    player_dropout_median_frames: float = 50.0
    player_dropout_log_sigma: float = 1.0
    # Automatic tracking is off by a little; clean sample positions are not. 0.1 m per axis is a
    # mild, deliberately conservative error. It drifts smoothly (white noise averaged over ~1 s)
    # rather than jumping every frame: per-frame jitter inflated median player speed by ~0.5 m/s
    # and erased the walking-at-stoppages signal (0% of dead frames under 1 m/s vs 57% clean),
    # whereas the challenge ATD keeps it (players slow on 47% of untracked-ball frames).
    position_noise_sd_m: float = 0.1
    position_noise_smoothing_frames: int = 25
    # The sample games' ball is often glued to a player's feet (nearest player p10 0.09 m, within
    # 1 m on ~53% of tracked frames); the challenge's automatic ball is not (p10 0.44 m, 28.5%)
    # and wobbles 2-3x more frame to frame. Ball error = a smoothly drifting offset plus
    # per-frame jitter, sized so those statistics match the real match
    # (scripts/compare_harness_realism.py).
    ball_noise_drift_sd_m: float = 0.8
    ball_noise_jitter_sd_m: float = 0.003
    # The real ball barely wobbles (|acceleration| p50 0.01 m/frame^2), so its error must curve
    # gently: averaged twice over this window, not once.
    ball_noise_smoothing_frames: int = 50
    # Sources whose ball is tracked on every frame (DFL/TRACAB) lose it the way an optical
    # tracker does, measured on the Metrica games: whole dead stretches lose it 96.8% of the time;
    # in live play ~1.27 gaps per minute cover 8.8% of frames — half are a few frames (brief
    # occlusions), half last seconds (heavy tail: 10% over 10 s).
    dead_stretch_ball_lost_share: float = 0.97
    live_ball_gaps_per_minute: float = 1.27
    live_ball_gap_short_share: float = 0.5
    live_ball_gap_long_median_s: float = 4.0
    live_ball_gap_long_log_sd: float = 1.2
    # Speeds are derived from positions over this centred window, as a tracker would smooth them.
    speed_window_frames: int = 5


DEFAULT_DEGRADE_CONFIG = DegradeConfig()


def degrade_labelled_game(game: labelled_games.LabelledGame, degrade_config: DegradeConfig,
                          seed: int) -> domain.TrackingData:
    tracking = (_remove_ball_like_optical_tracker(game.tracking, game.truth, degrade_config,
                                                  np.random.default_rng(seed + 5000))
                if game.is_ball_always_tracked else game.tracking)
    return degrade_tracking(tracking, degrade_config, seed)


def _remove_ball_like_optical_tracker(tracking: domain.TrackingData,
                                      truth: domain.LabelPerFrame, degrade_config: DegradeConfig,
                                      random_generator: np.random.Generator
                                      ) -> domain.TrackingData:
    is_ball_lost = np.zeros(len(truth), dtype=bool)
    for start, end in runs.true_runs(truth == domain.LABEL_DEAD):
        is_ball_lost[start:end] = (random_generator.random()
                                   < degrade_config.dead_stretch_ball_lost_share)
    live_frames = np.flatnonzero((truth == domain.LABEL_HOME) | (truth == domain.LABEL_AWAY))
    n_gaps = random_generator.poisson(degrade_config.live_ball_gaps_per_minute * len(live_frames)
                                      / (60 * config.FRAMES_PER_SECOND))
    for gap_start in random_generator.choice(live_frames, size=n_gaps, replace=False):
        is_short = random_generator.random() < degrade_config.live_ball_gap_short_share
        length = (int(random_generator.integers(1, 6)) if is_short
                  else int(config.FRAMES_PER_SECOND * degrade_config.live_ball_gap_long_median_s
                           * np.exp(random_generator.normal(0.0,
                                                            degrade_config.live_ball_gap_long_log_sd))))
        is_ball_lost[gap_start:gap_start + length] = True
    ball_norm = tracking.ball_norm.copy()
    ball_norm[is_ball_lost] = np.nan
    return dataclasses.replace(tracking, ball_norm=ball_norm)


def degrade_tracking(clean_tracking: domain.TrackingData, degrade_config: DegradeConfig,
                     seed: int) -> domain.TrackingData:
    random_generator = np.random.default_rng(seed)
    players_norm = clean_tracking.players_norm.astype(np.float64).copy()
    noise_sd_norm = np.array([degrade_config.position_noise_sd_m / clean_tracking.pitch_length_m,
                              degrade_config.position_noise_sd_m / clean_tracking.pitch_width_m])
    players_norm[:, :, :2] += _smooth_unit_noise(
        players_norm[:, :, :2].shape, degrade_config.position_noise_smoothing_frames,
        random_generator) * noise_sd_norm
    players_norm[_player_dropout_mask(players_norm.shape[0], players_norm.shape[1],
                                      degrade_config, random_generator)] = np.nan
    players_norm[:, :, 2] = _speed_from_positions_mps(players_norm, clean_tracking,
                                                      degrade_config.speed_window_frames)
    ball_norm = clean_tracking.ball_norm.astype(np.float64).copy()
    pitch_size_m = np.array([clean_tracking.pitch_length_m, clean_tracking.pitch_width_m])
    ball_error_m = (_twice_smoothed_unit_noise((len(ball_norm), 2),
                                               degrade_config.ball_noise_smoothing_frames,
                                               random_generator)
                    * degrade_config.ball_noise_drift_sd_m
                    + random_generator.normal(0.0, degrade_config.ball_noise_jitter_sd_m,
                                              (len(ball_norm), 2)))
    ball_norm[:, :2] += ball_error_m / pitch_size_m
    return dataclasses.replace(clean_tracking, players_norm=players_norm.astype(np.float32),
                               ball_norm=ball_norm.astype(np.float32))


def _smooth_unit_noise(shape: tuple[int, ...], smoothing_frames: int,
                       random_generator: np.random.Generator) -> NDArray[np.float64]:
    """White noise averaged along time (axis 0), then rescaled back to unit variance — averaging
    k samples shrinks the standard deviation by sqrt(k)."""
    white_noise = random_generator.normal(0.0, 1.0, shape)
    cumulative = np.concatenate((np.zeros((1, *shape[1:])), np.cumsum(white_noise, axis=0)))
    moving_average = (cumulative[smoothing_frames:] - cumulative[:-smoothing_frames]) \
        / smoothing_frames
    padded = np.concatenate((moving_average,
                             np.repeat(moving_average[-1:], smoothing_frames - 1, axis=0)))
    return padded * np.sqrt(smoothing_frames)


def _twice_smoothed_unit_noise(shape: tuple[int, int], smoothing_frames: int,
                               random_generator: np.random.Generator) -> NDArray[np.float64]:
    """Averaging twice gives a triangular kernel: the error drifts without frame-to-frame kinks.
    Rescaled to unit standard deviation empirically, since the double average has no simple
    closed-form shrinkage once padded."""
    once = _smooth_unit_noise(shape, smoothing_frames, random_generator)
    cumulative = np.concatenate((np.zeros((1, shape[1])), np.cumsum(once, axis=0)))
    twice = (cumulative[smoothing_frames:] - cumulative[:-smoothing_frames]) / smoothing_frames
    twice = np.concatenate((twice, np.repeat(twice[-1:], smoothing_frames - 1, axis=0)))
    return twice / twice.std(axis=0)


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
