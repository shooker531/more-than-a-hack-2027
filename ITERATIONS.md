# Iteration log

One entry per experiment, newest first — including the ones we revert. This feeds the technical
document's "approach" and "assets tried and dropped" sections.

## How an iteration works

1. Make one change (one idea per iteration).
2. `.venv/bin/python run.py` — writes the submission and must pass the validator.
3. Score on the validation games (harness: in progress).
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

## 001 — MVP rule-based state machine  (2026-11-08)  ⏳ PENDING VALIDATION

**Why:** the official baseline flickers (nearest player per frame, then a majority vote), has no
real dead-ball detection, and smears transitions by the vote window.

**Change:** first full pipeline (`run.py` → `mtah/`):
- dead ball = ball beyond the lines, players walking (< 1 m/s) for ≥ 2 s, set-piece tags until
  the restart kick, and before kick-off;
- control evidence = slow ball (< 7 m/s) with the nearer team's player within 2 m;
- possession switches only after the other team controls the ball for ≥ 0.5 s, backdated to when
  that control began; frames without evidence hold the last team.

**Validation:** pending the harness.

**Full-match sanity:** 345 possessions, mean 13.7 s (median 9.1 s), 48 under 2 s; D 25.8%,
H 44.5%, A 29.7%. The mean is close to the public window's 14.2 s.

**Public 5-min check:** valid file, frames aligned; spellF1 0.42 / macroF1 0.67 (baseline_smooth50:
0.33 / 0.62). Correctness check only, not a ranking.

**Commit:** see `experiments.csv`.

**Notes:** the ball is missing in long blocks (150 gaps, median 11 s, 18% of the match in gaps
≥ 30 s), and the MVP assumes no turnovers inside them. Inferring possession from player
movement during gaps is the most promising next step. An untracked ball also correlates with
stoppages (players slow on 47% of untracked frames vs 10% tracked): a candidate soft signal for
dead ball.

## 000 — Official baseline  (reference)

Nearest tracked player to the ball, centred majority vote ±50 frames (`baseline_smooth50`).
Full-match spellF1 **0.2433** / macroF1 0.6724 (EVALUATION.md). The floor to beat. It will also
be scored on the validation games once the harness exists, so 001+ can be compared fairly.
