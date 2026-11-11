# Iteration log

One entry per experiment, newest first, including rejected ones. Full numbers and configs are in
`experiments.csv`.

## Scoreboard

Validation spellF1 on the sample games (higher is better). Scores before the harness fix are not
comparable with those after it.

| # | Change | Validation spellF1 | Result |
|---|---|---|---|
| 005 | Touch evidence; switch on two opponent touches | **0.596** | ✅ kept |
| 004 | Long gaps seen in open play stay live | 0.526 | ❌ rejected |
| — | *Harness fix (002 re-scored: 0.528)* | — | — |
| 003 | End dead gaps at the restart | 0.533 | ❌ rejected |
| 002 | Ball missing ≥ 2 s = dead ball | 0.540 | ✅ kept |
| 001 | MVP rule-based state machine | 0.421 | ✅ kept |
| 000 | Official baseline | 0.339 | reference |

## How we iterate

1. **Diagnose first** on the sample games: check the errors are where you think they are.
2. Make one change.
3. Run `run.py`, then `run_validation.py`.
4. Log it: `scripts/log_experiment.py --iteration NNN --title "..."`. Uncommitted work is saved
   as `experiments/patches/NNN.patch`.
5. Keep it only if validation spellF1 beats the noise (intervals don't overlap). Otherwise
   discard the change; the patch keeps it for the write-up.

Full-match counts are sanity checks only. The public 5 minutes are for correctness checks only,
never for choosing versions.

---

## 005 — Touch evidence; switch on two opponent touches  ✅ KEPT

**Problem:** we matched only 47% of true possessions. The misses were short possessions with the
ball tracked. At turnovers the new team's player was right by the ball (median 0.4 m), but the
ball was fast (median 12 m/s), so our "slow ball = control" rule ignored it.

**Change:** a player within 1 m of the ball counts as a touch, at any speed. Possession switches
after two touches in a row by the other team (e.g. interception then pass), as well as after
0.5 s of control.

**Result:** 0.528 → **0.596**. Possessions predicted 2521 → 2836 (true 3399). 3 touches: 0.564;
1.5 m radius: 0.577.

**Next:** still missing ~560 possessions. Re-run the diagnosis on 005.

## 004 — Long gaps seen in open play stay live  ❌ REJECTED

**Problem:** 83% of wrongly dead frames are ball gaps during live play.

**Change:** a gap stays live if the ball was last seen > 3 m from every line and players never
walk for 2 s.

**Result:** 0.526 vs 0.528. Fouls, offsides and goals also stop play in open field, so location
alone hurts (0.513).

**Learned:** walking players are often just settled play, so walking is a weak stoppage signal.
Dead-ball work has hit diminishing returns.

## Harness fix — smooth position noise

The fake tracking noise jumped every frame, so players never looked like they were walking, and
the walking-stoppage rule never fired in validation. Noise now drifts smoothly. 002 re-scored at
**0.528**, the new reference.

## 003 — End dead gaps at the restart  ❌ REJECTED

**Problem:** 002 calls 5–9% of live frames dead.

**Change:** a long gap stays dead only until play restarts (set-piece tag, or players speed up).

**Result:** 0.533 vs 0.540. Only 0.7% of the wrongly dead frames were after a restart; 83% were
whole gaps in live play. Should have diagnosed first.

## 002 — Ball missing ≥ 2 s = dead ball  ✅ KEPT

**Problem:** the tracker mostly loses the ball when play stops, and the MVP ignored that.

**Change:** any ball gap of 2 s or more is dead ball.

**Result:** 0.421 → **0.540**. Dead-ball F1 0.82 → 0.94. Any threshold from 1 to 4 s works.

**Side effect:** full match is now 47% dead ball, probably too high (estimate ~34%).

## 001 — MVP rule-based state machine  ✅ KEPT

**Change:** first pipeline. Dead ball = ball out, players walking ≥ 2 s, set pieces, before
kick-off. A team gets possession after 0.5 s of control (player within 2 m of a slow ball).

**Result:** **0.421** vs baseline 0.339, winning on all 9 runs. Weaker than the baseline on dead
ball.

**Learned:** in the sample games the ball is missing on 92–100% of dead frames. Our match shows
the same pattern, which led to 002.

## 000 — Official baseline

Nearest player to the ball, smoothed over ±2 s. Full match 0.2433 (published floor);
validation 0.339.
