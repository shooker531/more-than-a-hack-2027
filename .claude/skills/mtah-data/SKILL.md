---
name: mtah-data
description: Core context for the More Than A Hack 2027 possession challenge (BAR-ATH) — the task, scoring, every provided dataset, file formats, frame alignment, the starter/ helpers, rules and known pitfalls. Load before writing any code that reads the match data, builds features, writes a submission, or interprets a score.
---

# MTAH 2027 — data and challenge context

Source of truth, in priority order: `evaluation/possession_metrics.py` (defines the metrics) >
`DATA.md` / `CHALLENGE.md` / `EVALUATION.md` / `RULES.md` / `FAQ.md` > this file. If they
disagree, they win — and fix this file.

## 1. The task in one paragraph

One match, **Barcelona (home, `H`) vs Athletic Club (away, `A`)**, tactical camera, **25 fps,
199,319 frames**. Output one label per frame: `H` home in possession (ball alive), `A` away in
possession (ball alive), `D` dead ball, `X` abstain. Only the two halves are graded (158,857
frames). Ranked on **spellF1** (possession-level), tie-broken by **macroF1** (frame-level).
**Published floor to beat: spellF1 0.2433** (`baseline_smooth50`, full match).

## 2. Hard working rule: keep the data out of the LLM

FAQ.md / the T&Cs: participants must not upload or disclose the provided videos, datasets **or any
extracts thereof** to any third-party hosted or cloud-based LLM. That includes this assistant.

- Write scripts that compute and print **aggregates** (counts, rates, scores, shapes, column
  names). Do not `cat`/`head` raw tracking rows, dump DataFrames of positions, print tag
  instances, or read frames of video into the conversation.
- Code, docs, scores from `run_evaluation.py`, and the public repo files are fine to read.
- If a debugging step genuinely needs raw values, have the user look at them locally.

## 3. Files and where they live

All data is gitignored under `data/` (or `$MTAH_DATA_DIR`). Never commit it — the submission repo
must be public and the data is not redistributable. Never hard-code paths: use `games.py`.

```python
import games
p = games.asset_paths()                    # tactical (default) — the one submissions index to
p["atd_meta"], p["atd_raw"], p["homography"], p["smart_tagging"], p["video"]
games.asset_paths(source="broadcast")      # no smart_tagging key
games.require(["atd_raw"])                 # same, but raises a readable error if missing
```

```
data/tactical/   BAR-ATH_tactical{.mp4, _atd.xml, _atd.txt, _homography.csv, _smart_tagging.xml}
data/broadcast/  BAR-ATH_broadcast{.mp4, _atd.xml, _atd.txt, _homography.csv}
```

Local status: all non-video files present; **both .mp4 files are missing** (not needed for a
submission).

Environment: `.venv` (uv, Python 3.11) with `requirements-starter.txt` (numpy, pandas,
kloppy==3.18.0, matplotlib, notebook). Run with `.venv/bin/python` or `source .venv/bin/activate`.
Add packages with `uv pip install --python .venv <pkg>`. The scorer itself needs only
numpy+pandas.

## 4. Frame alignment — the whole story

Everything on the tactical timeline is already aligned. No offsets to discover.

| source | index | → submission row |
|---|---|---|
| submission | `frame` 0…199318 | — |
| ATD | `frame_id` 1…199319 | `frame_id - 1` |
| Smart Tagging | seconds from video start | `round(seconds * 25)` |
| homography | `frame` 0…199318 | same |

`games.py` also has per-half `offsets` / `video_to_id` / `id_to_video`. Those map to the
**ground truth's internal IdFrame counter** — participants never need them. Ignore unless reading
GT meta.

### Graded frames

| span | video frames | graded |
|---|---|---|
| pre-match | 0 – 11,184 | no |
| **1st half** | **11,185 – 90,651** | yes (starts at kickoff) |
| halftime | 90,652 – 110,807 | no |
| **2nd half** | **110,808 – 190,197** | yes |
| post-match | 190,198 – 199,318 | no |

`games.half_video_blocks("BAR-ATH")` → `[(1, 11185, 90651), (2, 110808, 190197)]`.
**Trap:** 2nd half is graded from 110,808 but kicks off at **113,742** — those 2,934 frames are
all `D`. Write `X` on ungraded frames (free).

## 5. The datasets

### 5a. Automatic Tracking Data (ATD) — the core input

Metrica FIFA/EPTS export, **automatic, uncorrected**.

- **Raw line format:** `<frame_id>:<slot1>;<slot2>;...;<slot46>:<ball>`; each group `x,y,speed`;
  missing = `NaN`. Split on `:` → 3 parts; split players on `;`.
- **Coordinates normalised 0–1.** Pitch is not square — **scale to metres before computing
  distances** (`games.GAMES["BAR-ATH"]["pitch"]` = 105.3 × 68.0 m, measured; ATD metadata says
  105 × 68). Speed `s` is m/s (ball speed often NaN).
- **46 track slots**, not players. Slots 1–24 → Barcelona, 25–46 → Athletic (from `teamId` in the
  XML; first `<Team>` in the XML is home). `ShirtNumber` = slot index, not a shirt. Slots appear,
  vanish, swap identity, occasionally duplicate. Mean ~20.5 tracked objects per graded frame.
  **Team membership is reliable-ish; identity is not** — features must be collective/team-level.
- **Ball present on 56.9% of all frames, 59.3% of graded** → missing on **41% of graded frames**,
  disproportionately in duels, crowded boxes, aerial balls. The single most important fact.
- **Contains no ball-status (alive/dead) and no ball-owner fields.** Those exist only in the
  organisers' corrected tracking that produced the GT.
- Loading:
  - Fast/streaming, numpy only: copy the parser pattern in `starter/baseline.py`
    (`team_of_each_track`, `nearest_player_labels`). ~1–2 min per pass.
  - kloppy: `starter/load_atd.load()` → `TrackingDataset`; `.to_df()` gives `frame_id, ball_x,
    ball_y, {n}_x, {n}_y, {n}_s`. Full match through kloppy costs minutes and GBs — **parse once
    and cache to parquet/npy** (gitignored location, e.g. `data/cache/`).
  - `load_atd.check_alignment_raw()` verifies 199,319 contiguous ids from 1.

### 5b. Smart Tagging — automatic tactical phases

SportsCode XML, **638 instances**, automatic (not GT, has FP/FN).

- Load: `starter/load_smart_tagging.load()` (stdlib) → dicts with `code, start_s, end_s,
  start_frame, end_frame` (end is **exclusive**), `tags{Team, Half, Type, Side, ...}`.
  Or `kloppy.sportscode.load(path).to_df()` (columns `timestamp`, `end_timestamp`, `code`, and
  title-case tag columns).
- **Case:** codes UPPER (`SET PIECES`), tag values Title (`Kick Off (Start)`), tag columns Title
  (`Team`). Match case-insensitively.
- Codes: PLAYERS IN THE BOX 97, BALL IN FINAL THIRD 93, BALL IN THE BOX 72, PROGRESSION 66,
  DEFENDING IN MIDDLE THIRD 66, SET PIECES 59, CREATING CHANCES 30, DEFENDING IN DEFENSIVE THIRD
  30, LONG BALL 29, ATTACKING TRANSITION 24, DEFENSIVE TRANSITION 24, BUILD UP 22, DEFENDING IN
  ATTACKING THIRD 22, GOALS 4.
- `Team`: `Barcelona` (→ H) 314, `Atletic Club` (→ A, note spelling) 288, **`N/A` 36 — includes
  every throw-in**. Handle N/A explicitly.
- Set piece `Type`: Throw-in 26, Corner Kick 12, Goal Kick 10, Free Kick 5, Kick Off (After Goal)
  4, Kick Off (Start) 2.
- `Half` is a string (`1st Half`/`2nd Half`); `Max Players in the box` is a string (`7+`).
- **Intervals are windows, not instants:** set-piece tags open ~3.5 s before the actual restart.
  Use tags for *roughly when*, tracking for *exactly when*.
- **Phases ≠ possessions:** a Build Up may cover part of, one, or two possessions. Evidence, not
  labels.

### 5c. Homography

CSV `frame,h00..h22,reprojection_error,valid`, one row per video frame. Maps **normalised pitch
(0–1) → image pixels** (feed ATD coords directly; don't convert to metres first).
`starter/load_homography.load()` → `(matrices (n,3,3), reprojection_error, valid)`; `project`,
`unproject`, `pitch_corners`. Tactical: valid everywhere, median reprojection error 0.24 in play.
Only useful together with the video.

### 5d. Video (not present locally)

Tactical 1920×1080, 25 fps, 199,319 frames — the frame-index authority. If a decode gives a
different count, the decode is wrong; don't pad/trim. Allowed for watching/debugging and for
algorithmic CV features; **not** for turning observations into hand labels or frame-specific
rules.

### 5e. Broadcast recording

Own ATD (164,488 frames, 51 slots, ball 50.5%) and homography (valid 65.8%), **not aligned** to
tactical, cuts/replays, no mapping provided. Default: ignore. A ~164k-row submission means you
indexed broadcast by mistake.

## 6. Ground truth — what you're actually predicting

- Built **mechanically from corrected tracking**: `D` where corrected tracking marks the ball
  dead; otherwise the team the corrected tracking names as owner. No smoothing, no hand edits.
  So the target is "what corrected tracking calls dead / owning", not a human's view.
- Three graded states only. No "contested"/"transition". Loose/contested ball → stays with last
  team in control. A change needs **controlled** possession by the other team — not a touch, a
  deflection, or an aerial duel without follow-up control.
- Dead-ball ownership is not asked: dead = `D`, always.
- A **possession = maximal run of H or of A**. `H H D D H H` = two possessions.
- **Public GT:** `evaluation/gt/BAR-ATH_possession_gt_public.csv` — full length, real labels only
  on frames **11,185–18,684** (first 5 min; H 3,454 · A 2,577 · D 1,469; 17 possessions, 21
  segments, mean 14.2 s); everything else `X`.
  **Use only to check pipeline correctness. Not for training, not for tuning (a rule).** The
  smoothing window that wins on it loses on the full match.

## 7. Scoring — what moves the number

- **spellF1 (ranking):** possessions matched 1-to-1 to GT by temporal IoU; F1 averaged over IoU
  0.3–0.7. Constant guess = 0. 1 frame late free; 3 frames 0.98; 8 frames 0.89; **1 s late 0.65**.
  Noise band on full match ~0.04.
- **macroF1 (tie-break):** mean per-class F1 of H/A/D per frame. Floor ~0.16. Noise ~0.01.
- **Fragmentation kills spellF1:** flicker creates spurious possessions (raw baseline: 137
  predicted segments vs 21 true in the public window).
- **`X` handling:** macroF1 counts it as a miss. spellF1 **forward-fills X with your last label in
  that half** (back-fills at half start). So X inside a possession is free; X across a hand-over
  carries the old label past it — guess at transitions.
- Baselines (nearest player to ball, centred majority vote ±w frames; ball missing → D):

  | | public spellF1 | full spellF1 | full macroF1 |
  |---|---|---|---|
  | raw | 0.038 | 0.0486 | 0.6532 |
  | smooth25 | 0.287 | 0.1996 | **0.6752** |
  | smooth50 | 0.326 | **0.2433** | 0.6724 |
  | smooth125 | **0.424** | 0.2242 | 0.6356 |

## 8. Tooling commands

```bash
.venv/bin/python starter/baseline.py [--window 0 50 ...]   # regenerate baselines
.venv/bin/python evaluation/validate_submission.py evaluation/results/<name>.csv
.venv/bin/python evaluation/run_evaluation.py [--bootstrap 1000] [--no-detail]
.venv/bin/python starter/load_atd.py | load_smart_tagging.py | load_homography.py   # self-checks
```

- Submissions go in `evaluation/results/*.csv` (gitignored except `baseline_*`). File name = board
  name.
- After scoring: `evaluation/reports/<name>_transitions.csv` lists every transition as
  `matched` / `missed` / `spurious` with frames and latency — the main error-analysis tool.
  `evaluation/scoreboard.csv` has full metrics (possession count, latency, per-class P/R, etc.).
- In Python: `sys.path.insert(0, "evaluation"); import possession_metrics as pm;
  pm.evaluate(pm.label_codes(gt_df), pm.label_codes(pred_df))`.
- `starter/utils.py`: `draw_pitch`, `plot_frame(dataset, i)`, `animate_frames(dataset, a, b)`
  (kloppy dataset). Notebooks `starter/01..03` walk through tagging, ATD, first submission.

## 9. Submission contract

CSV, header exactly `frame,label`, **exactly 199,319 rows**, `frame` = 0…199318 in order, labels
in `H A D X`, no blanks (write `X`). ~1.9 MB. Always validate the exact file you upload.
Deliverables (deadline **2026-11-17**): `<team_name>.csv`, a public repo that **regenerates the
identical file** (pin deps, deterministic — seed everything), and a short technical doc
(`<team_name>.pdf`/.md: approach, assets used/dropped, training data, possession rules
implemented, limitations, reproduction). No live leaderboard.

## 10. Rules that constrain design

- Labels must come from an algorithm. **No hand labels on this game in any amount** — not for
  training, validation, or correction. Hand labels on *other* games are fine (document them).
- No human-produced data about this game from any source (event data, corrected tracking, etc.),
  and no private GT.
- No rules/constants tied to specific frames of this game learned by watching it.
- Unsupervised / self-supervised / algorithmic pseudo-labels on this game: allowed.
- Improving or replacing the tracking algorithmically: allowed.
- Anything used must be accessible to organisers and explained in the doc. Teams of 3–5.

## 11. Pitfalls checklist

- [ ] Off-by-one: ATD `frame_id - 1`; Smart Tagging `end_frame` exclusive.
- [ ] Distances in metres, not normalised units.
- [ ] Ball NaN on 41% of graded frames — never assume it exists.
- [ ] Track slots aren't people; don't build per-player features across time without re-linking.
- [ ] Smart Tagging `N/A` team; case-insensitive matching; windows open ~3.5 s early.
- [ ] 2nd-half frames 110,808–113,741 are graded `D`.
- [ ] Don't tune on the public 5 minutes; check correctness only. Use `--bootstrap` before
      believing a gap.
- [ ] Count your possessions — far more than GT-like (~mean 14 s) means flicker.
- [ ] Cache parsed ATD; don't re-parse 199k lines through kloppy every run.
- [ ] Never print raw data extracts into the LLM conversation (§2).
