# Iteration log

One entry per experiment, newest first, including rejected ones. Full numbers and configs are in
`experiments.csv`.

## Scoreboard

Validation spellF1 on the sample games (higher is better). Scores are only comparable within
the same harness version (marked by the italic rows).

| # | Change | Validation spellF1 | Result |
|---|---|---|---|
| 012 | Train on 7 extra Bundesliga matches (DFL open data) | **0.629** (vs 0.618) | ✅ kept |
| — | *Harness: 10 matches (3 Metrica + 7 DFL); 009 re-scored 0.618, baseline 0.312* | — | — |
| 011 | Smart Tagging phase features | 0.647 (vs 0.642) | ❌ rejected |
| 010 | Extra shape features (hull, lines, half share, ball position) | 0.644 (vs 0.642) | ❌ rejected |
| 009 | Learned possession model + Viterbi (leave-one-game-out) | **0.642** | ✅ kept |
| 008 | Recalibrate contest margin (1.0 m) and touch radius (1.5 m) | 0.578 | ✅ kept |
| — | *Harness fix: realistic ball error (006: 0.552, 005: 0.509, baseline: 0.330)* | — | — |
| 007 | Viterbi decoder (hand-set scores) | 0.643 (vs 0.658) | ❌ rejected |
| — | *Truth fix: controlled possession only (005: 0.638, 006: 0.658)* | — | — |
| 006 | No evidence when the ball is contested | 0.610 | ✅ kept |
| 005 | Touch evidence; switch on two opponent touches | 0.596 | ✅ kept |
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
5. Keep it only if validation spellF1 beats the noise: intervals don't overlap, or it wins on
   (nearly) every game/seed run against the previous version. Otherwise
   discard the change; the patch keeps it for the write-up.

Full-match counts are sanity checks only. The public 5 minutes are for correctness checks only,
never for choosing versions.

---

## 012 — Train on 7 extra Bundesliga matches  ✅ KEPT

**Problem:** the model learned from only 3 games, its biggest weakness. Extra noisy copies of
the same games didn't help (1 → 3 → 5 copies: 0.642 → 0.647 → 0.647): they add no new football.

**Change:** added the IDSSE / DFL open data (7 Bundesliga matches, TRACAB tracking at 25 Hz,
CC-BY 4.0). Its tracking carries per-frame ball possession and ball status, like the corrected
tracking behind the challenge's answer key. That flag flips on every touch (37% of spells under
2 s), so the controlled-possession rule (≥ 2 s) is applied: the result matches the public answer
key's shape (median 6.9 s vs 7.6 s). The ball is removed the way our tracker loses it (dead
stretches, plus live gaps calibrated on Metrica); the degraded data then matches our real match on
the realism statistics. Validation now covers all 10 matches, leave-one-game-out.

**Result:** pure held-out first — 009, trained on Metrica only, scored 0.57–0.64 on the unseen
Bundesliga matches, so it already transferred. Training on all other matches: 0.618 → **0.629**,
better on 22 of 30 runs and 8 of 10 matches (Bundesliga matches up to +0.030). Baseline 0.312.
Public 5 minutes (sanity only): 0.497 vs 0.447 for 009, 16 of 17 possessions.

**Full-match sanity:** 268 possessions, median 7.8 s, mean 12.5 s.

## 011 — Smart Tagging phase features  ❌ REJECTED

**Idea:** build-up / progression tags agree with the possessing team ~88% of the time on our
match (other phases ~64%), so give them to the model as two features.

**Change:** fake phase tags for the sample games, matched to the real tags' coverage, team
agreement, length and alignment with possession changes (all measured on our match without
labels); two features, +1 home / −1 away / 0 untagged.

**Result:** first version 0.652 (7/9 runs), but its fake tags lined up with possession changes
more neatly than the real ones. Matched to real alignment: **0.647 vs 0.642, 6/9 runs** — noise.

**Learned:** the tags are ~19 s windows over several possessions: they say who is attacking,
which the tracking already shows, not when possession changes. The first, too-clean version
would have looked like a gain — a reminder to check simulated inputs against the real ones.

**Code:** `experiments/patches/011.patch`.

## 010 — Extra shape features  ❌ REJECTED

**Idea:** defending teams are compact and deep, attacking teams stretched and high. Added convex
hull area (outfield players), defensive line (2nd-deepest player), press height (top 3 players),
share of players in the opponent's half, and ball position along the pitch.

**Result:** 0.644 vs 0.642; better on only 5 of 9 runs. Noise.

**Learned (feature importance on held-out games):** ball evidence dominates (shuffling it costs
21–25% frame accuracy). Shape does matter (~5–6% together) but the existing average position,
width and length already carry it; defensive line and press height add nothing new. Shape helps
say *who* has the ball mid-possession, not *when* it changes, and spellF1 scores the changes.

**Code:** `experiments/patches/010.patch`.

## 009 — Learned possession model + Viterbi  ✅ KEPT

**Problem:** the remaining errors were spread thin across many small causes, a sign the
hand-written rules were near their ceiling.

**Change:** a gradient-boosted model predicts P(home has the ball) on every live frame from 34
features: ball evidence (distances, touches, control, ball speed) plus team structure and
movement (how far up the pitch each team is, forward speed, width, length, pressing), averaged
over ±0.25 s, ±1 s and ±2.5 s. A Viterbi decoder turns the probabilities into possessions, with
one switch cost. Dead ball still comes from the rules. Trained on the sample games only;
validation is leave-one-game-out, so no game is scored by a model that saw it.

**Result:** 0.578 → **0.642** (intervals don't overlap), better in every game. Team structure
and movement are worth about +0.04 on their own (ball features only: 0.604). Public 5 minutes
(sanity only): 0.447 vs 0.332 for 006, with exactly 17 of 17 possessions.

**Next:** game 3 gains least and the model under-predicts possessions there (260 vs 383).
Smart Tagging build-up/progression tags as a feature, with realistic fake tags for the samples.

## 008 — Recalibrate contest margin and touch radius  ✅ KEPT

**Problem:** both settings were tuned on the old harness, where the ball sat unrealistically
close to players.

**Change:** contest margin 0.5 → 1.0 m (control and touch), touch radius 1.0 → 1.5 m.

**Result:** 0.552 → **0.578**, better on 8 of 9 runs. Predicted possessions now match the truth
(2818 vs 2802). Margins alone at 1.0 m: 0.571 (9/9 runs); 2.0 m: 0.563.

## Harness fix — realistic ball error

**Problem:** 006 barely beat the baseline on the public 5 minutes (0.332 vs 0.326) despite
doubling it on validation. Comparing statistics (no labels) showed why: in the sample games the
ball is glued to players' feet (within 1 m of a player 43–54% of the time vs 28.5% on our match),
so touch evidence fired twice as often as it really does.

**Change:** degraded tracking now adds ball error: a gently drifting 0.8 m offset plus small
jitter, sized to match the real match. `scripts/compare_harness_realism.py` checks this.

**Result:** scores drop to realistic levels: 006 0.658 → 0.552, baseline 0.356 → 0.330. The
ranking holds (006 > 005 0.509 > no touch rule 0.526), so our iterations still help.

## 007 — Viterbi decoder  ❌ REJECTED

**Problem:** the rules combine evidence badly: possessions are missed even with 2+ touches, and
one long ball gap always forces dead ball.

**Change:** each frame scores H, A and D from the evidence; every change of state has a cost;
Viterbi picks the best sequence per half. Replaces all hard thresholds with one trade-off.

**Result:** best 0.643 vs 0.658 for the rules (initial guesses 0.631). Scoring touches once per
touch instead of per frame helped (0.631 → 0.643); lowering the switch cost added false
possessions; lowering the control weight made things worse.

**Learned:** the rules encode order ("two touches in a row with no opponent touch between"),
which adding up per-frame scores cannot express. With hand-set scores Viterbi is a cruder
version of the rules. Worth revisiting once a learned model supplies calibrated per-frame
probabilities. A semi-Markov version would hit the same limit, so not tried.

**Code:** `experiments/patches/007.patch` (applies on top of 006).

## Truth fix — controlled possession only

**Problem:** the sample truth counted every "won it, lost it straight away" as a possession
(26% of possessions under 2 s), which CHALLENGE.md says is not a change of possession.

**Change:** a new team's spell only counts if it shows control: a completed pass, a carry, a
shot, a set piece, or keeping the ball for 2 s of live play. Otherwise it merges into the
previous team's possession.

**Result:** the truth now looks like the public answer key (under 2 s: 13% vs 12%; median 7.3 s
vs 7.6 s). Re-scored: 005 0.638, **006 0.658** (wins 9/9 runs vs 005), baseline 0.356.
Predicted possessions are now close to true (2644 vs 2802), so most of the "missing
possessions" were truth artefacts.

## 006 — No evidence when the ball is contested  ✅ KEPT

**Problem:** 32% of our predicted possessions matched nothing. The biggest cause (42%) was giving
the ball to the wrong team in duels and tight marking, where the defender is often nearest.

**Change:** a team only gets evidence when its player is at least 0.5 m nearer the ball than
any opponent. Otherwise the ball counts as contested and stays with the last team.

**Result:** 0.596 → **0.610**. Small but consistent: better on 8 of 9 runs. A 1.0 m margin was a
coin flip (6/9); 1.5 m was worse (0.587).

**Next:** done — see the truth fix above.

## 005 — Touch evidence; switch on two opponent touches  ✅ KEPT

**Problem:** we matched only 47% of true possessions. The misses were short possessions with the
ball tracked. At turnovers the new team's player was right by the ball (median 0.4 m), but the
ball was fast (median 12 m/s), so our "slow ball = control" rule ignored it.

**Change:** a player within 1 m of the ball counts as a touch, at any speed. Possession switches
after two touches in a row by the other team (e.g. interception then pass), as well as after
0.5 s of control.

**Result:** 0.528 → **0.596**. Possessions predicted 2521 → 2836 (true 3399). 3 touches: 0.564;
1.5 m radius: 0.577.

**Next:** still missing ~560 possessions (mostly a truth artefact, see the truth fix).

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
