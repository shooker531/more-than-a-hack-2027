"""Every threshold the pipeline uses, in generic units (metres, seconds, m/s), with its reasoning.

None of these are tied to frames of the match, and none were chosen against the public ground
truth. Values come from football knowledge and from match-wide aggregate distributions of the ATD
(scripts/data_profile.py). Selection between alternatives happens on external validation games.
"""

from __future__ import annotations

from dataclasses import dataclass

FRAMES_PER_SECOND = 25


def _seconds_to_frames(seconds: float) -> int:
    return int(round(seconds * FRAMES_PER_SECOND))


@dataclass(frozen=True)
class PipelineConfig:
    # --- ball kinematics -------------------------------------------------------------------
    # The ATD ball speed channel is 92% NaN, so speed is derived from positions. Positions jitter
    # frame to frame, so displacement is measured across a short centred window.
    ball_speed_window_frames: int = 5

    # --- dead ball -------------------------------------------------------------------------
    # Ball beyond the line by more than ~1 m (0.01 of pitch length) is out of play. A smaller
    # margin would fire on tracking jitter for balls rolling along the line.
    out_of_pitch_margin_norm: float = 0.01
    # Players walk (~1 m/s) at stoppages and jog or sprint in play. The median over tracked
    # players, averaged over 1 s, so one sprinter or one frozen track does not decide it.
    stoppage_player_speed_mps: float = 1.0
    player_speed_smoothing_frames: int = _seconds_to_frames(1.0)
    # Slow spells shorter than this happen in open play (a keeper holding the ball while the team
    # resets), so a stoppage must last at least this long.
    stoppage_min_duration_frames: int = _seconds_to_frames(2.0)
    # A visibly fast ball is in play regardless of how the players are moving.
    live_ball_min_speed_mps: float = 6.0
    # The tracker mostly loses the ball when play stops (out of play, held for a restart). On the
    # sample games, untracked-ball gaps of >= 2 s are 71-85% dead time, while gaps under 1 s are
    # almost all live (brief occlusions). The challenge ATD shows the same stoppage signature
    # (players slow on 47% of untracked frames vs 10% tracked). A gap this long counts as dead.
    ball_gap_dead_min_duration_frames: int = _seconds_to_frames(2.0)
    # Clean-up: drop dead runs shorter than this; absorb live runs shorter than this between two
    # dead runs (a restart immediately followed by another stoppage is rare).
    dead_min_duration_frames: int = _seconds_to_frames(1.0)
    live_min_duration_between_dead_frames: int = _seconds_to_frames(1.0)
    # Smart Tagging set-piece windows open a few seconds before the restart (~3.5 s on the
    # kick-offs, see starter/01_smart_tagging.ipynb). The restart is the first fast ball movement
    # after the tag opens; without one, this default offset is used.
    set_piece_kick_min_speed_mps: float = 6.0
    set_piece_default_restart_offset_frames: int = _seconds_to_frames(3.5)
    set_piece_restart_search_frames: int = _seconds_to_frames(8.0)

    # --- control evidence ------------------------------------------------------------------
    # A player controls the ball when within this distance: a stride plus the ball's distance
    # from the feet while dribbling. The match-wide median nearest-player distance is 1.9 m.
    control_radius_m: float = 2.0
    # A faster ball is a pass or shot in flight, not under anyone's control.
    control_max_ball_speed_mps: float = 7.0
    # A player this close is playing the ball, whatever its speed: a leg's reach plus the
    # automatic tracker's ball error (the real ball sits >= 0.44 m from the nearest player 90% of
    # the time, so contact rarely looks closer than ~0.5 m). Tighter than control_radius_m
    # because fast balls pass near players without being touched.
    touch_radius_m: float = 1.5
    # A team only gets evidence when its nearest player is this much nearer the ball than any
    # opponent. On the sample games 42% of invented possessions handed the ball to the wrong
    # team: in duels and under tight marking the defender is often the nearest player. With the
    # real tracker's ball error, "nearer" needs a full metre to be trustworthy.
    control_contest_margin_m: float = 1.0
    touch_contest_margin_m: float = 1.0

    # --- possession switching (the "controlled possession" rule) ---------------------------
    # Evidence frames of one team separated by holes up to this long form one control episode.
    control_episode_max_gap_frames: int = _seconds_to_frames(0.2)
    # An opposing episode must last this long to take possession: a touch or deflection is
    # shorter, receiving and controlling the ball is longer.
    possession_switch_min_episode_frames: int = _seconds_to_frames(0.5)
    # ...or the other team touches the ball this many times in a row with no touch by the
    # holding team in between: an interception then a pass, or a completed pass — CHALLENGE.md's
    # own proxy for control. One touch alone (a deflection, a duel) never switches possession.
    possession_switch_consecutive_touches: int = 2



DEFAULT_CONFIG = PipelineConfig()
