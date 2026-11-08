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
