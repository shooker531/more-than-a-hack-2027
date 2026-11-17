# Beyond XI: ball possession labelling

More Than A Hack 2027, Phase 1. Submission file: `Beyond_XI.csv`. Code: this repository, which
regenerates the file exactly (Section 8).

## 1. Summary

We label each frame as home possession (`H`), away possession (`A`) or dead ball (`D`) in three
steps: rules decide when the ball is dead, a gradient-boosted model estimates which team has the
ball on every live frame, and a Viterbi decoder turns those estimates into clean possessions.
The model is trained only on public matches from other competitions, made to look like the
challenge's automatic tracking. No label from the challenge match was used to train, tune or
choose any part of the system.

On ten public matches, each scored by a model that never saw it, our pipeline reaches a spellF1
of **0.613**, against **0.309** for the official baseline under the same conditions.

## 2. Method

### 2.1 Dead ball (rules)

A frame is `D` when any of these holds:

- the ball is tracked more than about 1 m beyond a touchline or goal line;
- the ball has been untracked for 2 s or more;
- the players' median speed, averaged over 1 s, stays below 1 m/s for 2 s or more, unless the
  ball is visibly moving fast;
- a Smart Tagging set-piece tag is open and the restart kick has not yet happened (the restart is
  the first fast ball movement after the tag opens);
- the half has not yet kicked off.

The second rule does most of the work. The challenge tracker mostly loses the ball when play
stops: on the public sample matches the ball is missing on 97% of dead frames but only 9% of live
ones, and the challenge match shows the same pattern in how players move when the ball is lost.

### 2.2 Who has the ball (gradient boosting)

For every live frame, a model outputs the probability that the home team has the ball. It uses
34 measurements, each written as home minus away so the model treats both teams alike:

- **Ball evidence.** Each team's nearest-player distance to the ball and their difference; ball
  speed (from positions); a *touch* signal, a player within 1.5 m of the ball at any speed; a
  *control* signal, a player within 2 m of a ball moving slower than 7 m/s. Touch and control
  only count for a team whose player is at least 1 m nearer the ball than any opponent, so duels
  and tight marking give no credit to either side.
- **Team structure and movement,** measured in each team's attacking direction (worked out per
  half from where each team's players stand, without labels): how far up the pitch the team's
  centre is and how fast it is moving forward, the team's width and length, its average player
  speed, and how many of its players are within 5 m of the ball.
- **Context.** Averages of these over ±0.25 s, ±1 s and ±2.5 s around the frame.

The model is scikit-learn's `HistGradientBoostingClassifier` (300 trees, learning rate 0.05,
31 leaves, L2 regularisation 1.0). Gradient boosting builds many small decision trees in turn,
each one correcting the errors of the trees before it. We chose it because it is strong on tables
of measurements like ours, handles missing values natively (the ball is often untracked), needs
little tuning, trains in seconds and is fully deterministic with a fixed seed.

### 2.3 From probabilities to possessions (Viterbi decoding)

Per-frame probabilities flicker, and the ranking metric punishes flicker hard: every brief flip
creates a false possession. Inside each stretch of live play, a two-state Viterbi decoder finds
the sequence of home and away labels that best fits the model's log-probabilities, while paying
a fixed cost every time possession changes. A handful of uncertain frames cannot pay for a
switch; a sustained shift in the evidence can, and the switch lands where the evidence turns
rather than after a fixed delay. The cost is 20, chosen on our validation matches (costs of 5, 10,
20 and 40 scored 0.620, 0.635, 0.642 and 0.643; 20 is where the gains level off). In practice, the
model must favour the other team at 90% confidence for about a third of a second, or at 70% for
about a second, before possession switches. After a dead stretch, either team may start the next possession
at no extra cost. We chose Viterbi because it is the standard, exact way to turn noisy per-frame
probabilities into a sequence of states, it has a single setting, and its behaviour is easy to
explain.

### 2.4 Alternatives we considered

| Option | Outcome |
|---|---|
| Hand-written rules for who has the ball (nearest player, control duration, two touches in a row) | Our first approach and the basis of iterations 001 to 008. Replaced when the learned model beat it on held-out matches (0.578 to 0.642). |
| Majority-vote smoothing, as in the official baseline | Places changes late and still flickers. |
| Viterbi with hand-set evidence scores | Worse than the rules (0.643 vs 0.658). Adding up per-frame scores cannot express order, such as two touches in a row. It only works with learned probabilities. |
| Neural sequence models (temporal convolutions, transformers) | Need far more labelled matches than our ten; also harder to make exactly reproducible in the time available. |
| Hidden semi-Markov decoding (modelling possession length) | Not pursued: our tests showed the decoder was not the bottleneck (Section 5). |
| Ball detection from the video | Not pursued. The tracker loses the ball mostly during stoppages, which our dead-ball rules already handle, and most of our remaining errors occur while the ball is tracked, so recovering the ball from video would mainly help where we are already right. A video pipeline was also too large for the time available. |

## 3. Data

| Asset | Used? | Notes |
|---|---|---|
| Tactical tracking (ATD) | Yes | All model inputs; the missing ball drives dead-ball detection |
| Smart Tagging set pieces | Yes | Dead time before restarts; kick-off timing |
| Smart Tagging phases of play | Tested, dropped | Their team agrees with possession about 88% of the time, but each tag spans about 19 s and several possessions; no measurable gain (0.647 vs 0.642) |
| Tactical and broadcast video | No | Available, but not used (see Section 2.4) |
| Homography | No | Only needed to project tracking onto the video |
| Broadcast tracking | No | Runs on a different timeline from the tactical frames we label |

## 4. Training and validation on other matches

The rules forbid labels on the challenge match, so the model learns from public matches where
possession is known, and every change is tested on them too.

| Source | Matches | Possession truth | Use |
|---|---|---|---|
| Metrica Sports sample data | 3 anonymised sample matches | Derived from their human-annotated events | Training and validation |
| IDSSE / DFL open data | 7 matches, Bundesliga and 2. Bundesliga, season 2022/23 | Per-frame `BallPossession` and `BallStatus` in the TRACAB tracking | Training and validation |
| SkillCorner open data | 20 A-League matches, season 2024/25 | Per-frame team possession | Tested as training data, not used in the submission (Section 6) |

The seven DFL matches: 1. FC Köln v FC Bayern München, VfL Bochum 1848 v Bayer 04 Leverkusen
(Bundesliga), and Fortuna Düsseldorf v 1. FC Nürnberg, SSV Jahn Regensburg, FC St. Pauli,
F.C. Hansa Rostock and 1. FC Kaiserslautern (2. Bundesliga). Full credits are in Section 10.

**Controlled possession.** In both sources possession flips on every touch (in the DFL data, 37%
of spells last under 2 s). CHALLENGE.md says a touch without control is not a change of
possession, so a new team only takes over if it keeps the ball for 2 s of live play, or, for the
event-based Metrica truth, completes a pass, carry, shot or set piece. Afterwards, possession
lengths match the public answer key's shape (median 6.9 to 7.3 s against 7.6 s).

**Making the public matches look like the challenge tracking.** Their tracking is cleaner than the
challenge's, so we degrade it: players get a slowly drifting 0.1 m position error and occasional
dropouts; the ball is smoothed over 0.84 s, as the challenge tracker smooths it, then offset by a
drifting error of about 0.8 m; and where a source tracks the ball on every frame (DFL), the ball is
removed on dead stretches and in short live gaps, at rates measured on the Metrica matches. We
check the result against the real match using statistics that need no labels (share of frames
with the ball tracked, ball-to-player distances, how often touch and control fire, how the ball
changes direction, player speeds at stoppages; `scripts/compare_harness_realism.py`). These checks
caught three unrealistic assumptions during the project, each of which had been flattering some
versions.

**Validation protocol.** Leave-one-game-out over the ten matches: each match is scored by a model
trained on the other nine, with three random degradations per match, using the organisers' own
scoring code. We kept a change only if it improved results consistently across matches, not just
on average.

## 5. Results and analysis

| | spellF1 on the ten public matches, leave-one-game-out |
|---|---|
| Official baseline (nearest player, ±2 s vote) | 0.309 |
| **Our submission** | **0.613** (Metrica 0.626, Bundesliga 0.608) |

Earlier comparisons that drove the main decisions (on earlier, less strict validation set-ups):
rules-only decoder 0.578 against the learned model 0.642 on three matches; training on Metrica
only 0.618 against Metrica plus DFL 0.629 on ten matches. The model trained on Metrica alone
scored 0.57 to 0.64 on the unseen Bundesliga matches, so it transfers across leagues and
tracking providers.

**Public five minutes.** Our submission scores spellF1 0.521 and macroF1 0.759 (16 predicted
possessions against 17), against 0.326 for the baseline. With 17 possessions this window is too
small to rank versions, and the rules forbid tuning on it; we used it only to check the file.

**Where the remaining errors are.** Replacing our dead-ball labels with the truth on the validation
matches lifts spellF1 only to 0.666; replacing our team labels lifts it to 0.930, so almost all
the headroom is in team possession. Of true open-play turnovers, 15% have a matching predicted
switch within 0.5 s, 42% within 2 s, and 46% have none within 5 s. Long spells on the wrong team
are rare (2% of live frames) and false switches are 13% of our predicted switches. The main error
is therefore short possessions that we miss entirely, where a team wins the ball and loses it
within a few seconds.

## 6. What we tried and dropped

Each idea was compared with our best version at the time, under the same validation set-up. The
score to beat changes between rows because we made the validation stricter during the project,
each time it turned out to be more forgiving than the real tracking.

| Idea | Score to beat | With the idea | Why it did not help |
|---|---|---|---|
| End dead-ball gaps at the estimated restart | 0.540 | 0.533 | Only 0.7% of false dead frames were after restarts |
| Keep ball gaps live when the ball was last seen in open play | 0.528 | 0.526 | Fouls, offsides and goals also stop play in open field |
| Team shape: convex hull, defensive line, press height | 0.642 | 0.644 | Already captured by team position, width and length |
| Smart Tagging phases as features | 0.642 | 0.647 | Tags span several possessions; they say who is attacking, not when possession changes |
| Switch penalty that depends on a learned switch model | 0.629 | 0.636 | Within noise; the decoder was not the bottleneck |
| Extra training weight around possession changes | 0.629 | 0.632 | Within noise |
| Shorter averaging windows | 0.629 | 0.620 | The longer windows steady the model |
| Sudden changes in ball direction near a player | 0.613 | 0.618 | Strong on clean sample balls, but the real tracker smooths most of it away |
| SkillCorner added at 15% of training weight | 0.613 | 0.609 | Broadcast tracking, with off-screen players estimated, looks unlike a tactical camera |
| SkillCorner added at full weight | 0.613 | 0.586 | As above |
| SkillCorner instead of the Bundesliga matches | 0.613 | 0.574 | The kind of tracking matters more than the number of matches |
| SkillCorner only | 0.613 | 0.567 | As above |

"Within noise" means the change did not improve results consistently across the ten matches.

Every iteration, including the numbers behind each decision, is in `ITERATIONS.md`.

## 7. Limitations

- **Short possessions are under-detected.** We predict about 6,800 possessions where the
  validation truth has about 9,000. We attacked this through the decoder, the training emphasis,
  the averaging windows and new ball features; none moved it beyond noise. The tracked ball is too
  noisy and too smoothed to reveal brief possessions reliably; better ball information, for example
  from the video, is the most likely route.
- **Dead ball is probably over-predicted.** About 47% of the challenge match is labelled `D`; an
  indirect estimate suggests nearer 34%. On validation, 4.3% of frames are called dead while live,
  mostly long ball gaps during live play.
- **Validation truth is an approximation** of the organisers' corrected-tracking labels, built from
  events or another provider's tracking with our controlled-possession rule.
- **Ten labelled matches from two providers.** Their degradation is simulated, though checked
  against the real match.

## 8. Reproduction

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements-mtah.txt
# put the challenge data in data/tactical/ (see data/README.md)
bash scripts/download_public_data.sh                   # Metrica and DFL public data, about 2.9 GB
.venv/bin/python run.py --output Beyond_XI.csv        # trains, labels and validates the file
.venv/bin/python run_validation.py                     # optional: leave-one-game-out validation
```

The same inputs always produce the same file (fixed seeds, single model). The DFL data is
downloaded from its Hugging Face mirror because kloppy's built-in download from Figshare did not
work for us; kloppy still parses the files.

## 9. Tools

Python 3.11 with numpy, pandas, scikit-learn and kloppy. Claude Code (Anthropic) was used as an AI
coding assistant during development, to help write code, run experiments and draft this document,
in accordance with the organisers' rules. It is not part of the submitted pipeline: `run.py` runs
without any AI model, and all design decisions were made and reviewed by the team.

## 10. Data credits

- **IDSSE / DFL open data.** Data owner: Deutsche Fußball Liga (DFL); collected by Sportec Solutions
  with TRACAB optical tracking. Licensed under CC-BY 4.0. Bassek, M., Weber, H., Rein, R. & Memmert,
  D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific
  Data*, 12, 195. https://doi.org/10.1038/s41597-025-04505-y. Data: https://doi.org/10.6084/m9.figshare.28196177
  (we used the mirror at https://huggingface.co/datasets/pysport/idsse-data).
- **Metrica Sports sample data,** https://github.com/metrica-sports/sample-data. Courtesy of
  Metrica Sports.
- **SkillCorner open data,** https://github.com/SkillCorner/opendata (MIT licence). Courtesy of
  SkillCorner. Tested during development; not used in the submitted model.
