# Iteration log

One entry per experiment, newest first — including the ones we revert. This feeds the technical
document's "approach" and "assets tried and dropped" sections.

## How an iteration works

1. Make one change (one idea per iteration).
2. `.venv/bin/python run.py` — writes the submission and must pass the validator.
3. `.venv/bin/python run_validation.py` — scores the pipeline and the official baseline on three
   public Metrica games (3 degradation seeds each, pooled spellF1 with a bootstrap interval).
4. `.venv/bin/python scripts/log_experiment.py --iteration NNN --title "..."` — appends the
   numbers to `experiments.csv`. Never type numbers in by hand.
5. Write the entry below, then commit as `iter-NNN: <title>`.

## Decision rule

- **Keep** a change only if it improves **spellF1 on the validation games** by more than the
  noise (non-overlapping bootstrap intervals, or a consistent gain across games).
- Full-match counts (possessions, durations, D share) are **sanity checks only**: they must stay
  football-plausible, but they cannot tell us which version is better.
- The public 5-minute window is for **correctness checks only** (alignment, file format). Never
  for choosing between versions — that is against the rules, and it picks the wrong version
  anyway (EVALUATION.md).

Entry template:

```markdown
## NNN — <title>  (YYYY-MM-DD)  ✅ KEPT | ❌ REVERTED | ⏳ PENDING
**Why:** the problem or hypothesis.
**Change:** what was done, which modules.
**Validation:** spellF1 before → after (games, interval).
**Full-match sanity:** possessions, mean duration, D share.
**Commit:** <hash>
**Notes:** what we learned, follow-ups.
```

---

## 003 — End gap dead stretches at the estimated restart  (2026-11-08)  ❌ REVERTED

**Why:** 002 over-predicts D (5–9% of validation frames called dead while live). Hypothesis: the
untracked live play after a restart, before the tracker reacquires the ball, was being labelled
dead.

**Change:** each long gap stayed dead only until an estimated restart — a set-piece restart
inside the gap, or the end of the last walking spell (median player speed < 1.5 m/s); with
neither, the whole gap stayed dead as in 002.

**Validation:** spellF1 0.533 [0.519, 0.547] vs 0.540 for 002. Speed-threshold sweep: tags only
0.540, < 1.0 m/s 0.539, < 1.5 0.533, < 2.0 0.539, < 2.5 0.541 — no variant beats noise.

**Commit:** 8608f67 (reverted in 61419e1).

**Notes — why it failed (measured on the sample games, 002 behaviour):** of the false-dead frames,
**83% are whole gaps with no dead time at all** (the ball lost for ≥ 2 s during live play), 13%
come from rules on tracked frames (walking-pace stoppage, set-piece windows, out of pitch), 3% are
live play before a stoppage inside a gap, and only **0.7% are the post-restart tail** this
iteration targeted. The hypothesis was wrong; the diagnosis should have come first.
→ Next: tell **live-play gaps from dead gaps**. Candidate cue: where the ball was last seen — a
ball that goes out of play is last seen near a touchline or goal line; a ball lost in live play
is usually last seen in open play.

## 002 — Sustained ball gaps count as dead ball  (2026-11-08)  ✅ KEPT

**Why:** 001's notes — the tracker mostly loses the ball when play stops, and the MVP ignored
that signal entirely.

**Change:** `dead_ball.py` adds one signal: any untracked-ball gap of ≥ 2 s is dead
(`ball_gap_dead_min_duration_frames`). On the sample games, gaps ≥ 2 s are 71–85% dead time while
gaps < 1 s are almost all live.

**Validation (3 games × 3 seeds, pooled):** spellF1 0.421 → **0.540** [0.525, 0.554], macroF1
0.543 → 0.858, alive/dead F1 0.816 → 0.942 (now above the baseline's 0.917). Baseline 0.339.
Threshold sweep was flat: 1 s 0.542, 2 s 0.540, 3 s 0.538, 4 s 0.537, 6 s 0.534 — so the gain is
not riding on a fragile number.

**Full-match sanity:** 350 possessions, mean 9.6 s (median 6.3 s), 56 under 2 s; D 47.2%,
H 33.0%, A 19.8%.

**Commit:** 84c8498

**Notes:**
- **D is now over-predicted.** Full match 47% vs an estimated ~34%; on the validation games
  predicted D exceeds truth by 3–9 points, with 5–9% of frames called dead while live, and
  almost no true dead time missed. Long gaps contain live play (the restart and first touches
  before the tracker reacquires the ball), and the stoppage / set-piece signals may add more.
  → 003 candidate: recover live play inside and around gaps (e.g. end the dead stretch at the
  restart, not at ball reacquisition).
- Still under-predicting possessions (2440 vs 3399 true): turnovers are being missed, especially
  inside live stretches where the ball is untracked. → candidate: player-movement evidence.
- Home share (33%) is well above away (20%) on the full match; worth checking it is football, not
  a bias in the evidence rule.

## 001 — MVP rule-based state machine  (2026-11-08)  ✅ KEPT

**Why:** the official baseline flickers (nearest player per frame, then a majority vote), has no
real dead-ball detection, and smears transitions by the vote window.

**Change:** first full pipeline (`run.py` → `mtah/pipeline.py`):
- dead ball = ball beyond the lines, players walking (< 1 m/s) for ≥ 2 s, set-piece tags until
  the restart kick, and before kick-off;
- control evidence = slow ball (< 7 m/s) with the nearer team's player within 2 m;
- possession switches only after the other team controls the ball for ≥ 0.5 s, backdated to when
  that control began; frames without evidence hold the last team.

**Validation (3 games × 3 seeds, pooled):** spellF1 **0.421** [0.407, 0.435] vs official baseline
0.339 [0.329, 0.349] — intervals don't overlap, and it wins on all 9 runs. But macroF1 0.543 vs
0.780 and alive/dead F1 0.816 vs 0.917: the baseline's dead-ball handling is better. It
under-predicts possessions (2324 vs 3399 true).

**Full-match sanity:** 345 possessions, mean 13.7 s (median 9.1 s), 48 under 2 s; D 25.8%,
H 44.5%, A 29.7%.

**Public 5-min check:** valid file, frames aligned. Correctness only.

**Commit:** 8e1a130 (pipeline code unchanged since 90f2209; output hash identical).

**Notes — the key finding for the next iteration:**
- In the sample games the ball is missing on 92–100% of dead frames and only 6–13% of live ones.
- The challenge's ATD matches the sample games on the stoppage signature: players are slow on
  47% of untracked-ball frames vs 10% tracked (samples: 52–56% vs 4–11%). Carried across, ~85%
  of the challenge's missing-ball frames are probably dead ball, and the match is probably ~34% D
  — the MVP labels only 25.8%.
- So **a sustained ball gap is likely our strongest dead-ball signal**, and the MVP deliberately
  ignores it. That is iteration 002.
- Caveat: the harness's ball is left undegraded, so if the challenge loses the ball in live play
  more than estimated, the harness will over-reward a gap → dead rule. Check the full-match D
  share stays plausible (~30–40%).

## 000 — Official baseline  (reference)

Nearest tracked player to the ball, centred majority vote ±50 frames (`baseline_smooth50`).
Full-match spellF1 **0.2433** / macroF1 0.6724 (EVALUATION.md). The floor to beat. Scored
alongside every iteration in `run_validation.py` (column `validation_baseline_spell_f1`):
0.339 on the validation games.
