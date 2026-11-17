---
name: coding-style
description: Coding style rules for this repository — functions over classes, pure and specific functions, verbose names, full typing, why-not-what docstrings, private-by-default helpers, no early-return escape hatches, single-type arguments. Load before writing or reviewing any Python in this repo.
---

# Coding style

The goal is a system that is **easy to reason about**: a reader can open a file, see its entry
points, and follow the data from input to output without having to simulate branches in their
head. When a rule here conflicts with getting something working, keep the rule and simplify the
design instead.

## 1. Functions over classes

- Default to plain functions.
- Use a class only when something genuinely **retains state across calls**.
- Immutable data groupings are fine and encouraged: `@dataclass(frozen=True)` or `NamedTuple` for
  config and records (e.g. `Config`, a `ControlEpisode`). They carry data, not behaviour — no
  methods beyond trivial derived properties.
- No inheritance hierarchies, no "manager"/"processor"/"handler" classes wrapping functions.

## 2. Pure, specific functions

- A function takes its inputs as arguments and returns its result. It does not mutate its
  arguments, read globals, or do I/O.
- I/O (reading data, caching, writing CSVs, printing) lives at the edges: `mtah/load.py`,
  `mtah/write.py`, `run.py`, `scripts/`. Everything in between is pure.
- One job per function. If the name needs "and", split it.
- Copy before modifying an input array (`out = mask.copy()`); never edit in place what was passed in.
- Deterministic: no unseeded randomness, no dependence on dict/set ordering for output, no
  wall-clock. The submission must regenerate byte-identically.

## 3. Naming — verbose beats ambiguous

- Name things for what they **mean**, not what they are: `is_ball_dead_per_frame`, not `mask`;
  `nearest_home_distance_m`, not `nh`; `control_episodes`, not `eps`.
- **Put units in names** whenever a value has one: `_m`, `_mps`, `_s`, `_frames`, `_norm`
  (normalised 0–1 pitch units). `stoppage_min_duration_frames`, `control_radius_m`.
- **Say which timeline an index is on**: `video_frame` (submission row, 0-based) vs `atd_frame_id`
  (1-based). Never a bare `frame` where both could be meant.
- Booleans read as questions: `is_`, `has_`, `was_`.
- Per-frame arrays end in `_per_frame` when it is not obvious from context.
- Functions are verbs: `detect_dead_ball`, `merge_evidence_into_control_episodes`,
  `find_restart_video_frame`.
- No `data`, `df`, `tmp`, `res`, `val`, `x`, `arr`, `helper`, `utils`, `process`, `handle`. Single
  letters only for maths that reads naturally (`dx`, `dy`) or tight loop indices.

## 4. Typing everywhere

- Every function signature has parameter and return annotations, including private ones.
- Start modules with `from __future__ import annotations`.
- numpy: `numpy.typing.NDArray[np.float64]`, `NDArray[np.bool_]`, `NDArray[np.int8]`. Define
  readable aliases once, in `mtah/domain.py`, and reuse them:
  `BoolPerFrame = NDArray[np.bool_]`, `TeamCodePerFrame = NDArray[np.int8]`, etc.
- State array shapes in the alias name or the docstring when they are not 1-D per frame
  (`players` is `(n_frames, n_slots, 3)`).
- Prefer `Literal["H", "A", "D", "X"]` / small `IntEnum`s over loose `str`/`int` for closed sets.

## 5. Docstrings explain decisions, not code

- Do **not** restate what the code or the name already says. `"""Compute the mean."""` on
  `compute_mean` is noise — leave it out.
- Do write what a reader **cannot** get from the code: why this approach, why this threshold, what
  was tried and rejected, what an edge case means in football terms, which convention applies
  (end-exclusive intervals, normalised vs metres, which timeline).
- Every module gets a docstring: its role in the pipeline and its public entry points.
- A short function with a precise name and types often needs no docstring at all.
- Threshold justifications live next to the value in `mtah/config.py`, not scattered in docstrings.

## 6. Private by default

- Anything not imported by another module is private: prefix with `_`.
- The public (un-prefixed) functions of a module are exactly its entry points — reading them should
  tell you how the module is used.
- Modules other than `run.py` and `scripts/` are libraries: no top-level side effects.

## 7. No escape-hatch logic

- No defensive early returns (`if x is None: return ...`, `if len(a) == 0: return []`) that create
  hidden alternative paths through a function.
- Make the input valid instead: validate **once at the boundary** (loading, config construction)
  and raise loudly there, so inner functions can assume well-formed inputs.
- Prefer formulations that handle edge cases naturally (vectorised ops on empty arrays, a loop
  that runs zero times) over special-casing them.
- Raise on impossible states; never silently return a default to "keep going".

## 8. One meaning per argument

A reader should never have to work out several ways a function behaves depending on what kind of
value was passed in.

- **Don't overload arguments.** No argument whose type or special value switches the code path
  (`int | str`, `x: int | None` where `None` means "do something else"). Two behaviours means two
  functions.
- **No "work it out yourself" defaults** (`data_dir=None`). Pass concrete values from the caller.
- **Absence must be real, named and documented.** If absence is a genuine state in the data (no
  team in control, ball not tracked), it is a named constant in `mtah/domain.py` (`NO_TEAM`,
  `NaN`) and the code treats it as a normal value. If absence only exists for convenience,
  remove it: raise, or restructure so the value is always present.

## 9. Pipeline entry point: one sequential dispatcher

- Each pipeline has **one entry-point function** (e.g. `main()` in `run.py`) that reads top to
  bottom as the list of processing steps, in order: load → features → detectors → decode →
  write → validate → report.
- The dispatcher only wires steps together: each line calls one step function and names its
  result. No logic, thresholds or branching inside it.
- Each step is a public function in its own module, taking explicit inputs and returning explicit
  outputs, so any step can be run, tested or swapped in isolation.
- Adding a processing step = one new module function + one new line in the dispatcher.

## 10. Import modules, not names

- Import the module and call through it: `from mtah import features` →
  `features.compute_frame_features(...)`; inside the package `from . import runs, domain`. The call
  site then says where a function lives, which makes it easy to trace.
- Third-party as usual: `import numpy as np`, `import pandas as pd`.
- Exceptions, for typing constructs only: names from `typing`, `numpy.typing` (`NDArray`) and
  `dataclasses` (`dataclass`) may be imported directly.
- Never name a variable or parameter after a module it sits beside (`features = features.…`).
  Rename the variable (`frame_features`, `pipeline_config`).

## 11. Structure and constants

- No magic numbers in logic. Thresholds go in `mtah/config.py` with units and a justification;
  physical/format constants (fps, pitch size, slot count) come from `games.py` or named module
  constants.
- Small modules by concern (`features`, `dead_ball`, `evidence`, `decode_*`). The pipeline order is
  visible in `run.py` as a straight sequence of calls.
- Vectorise with numpy where it stays readable; a clear Python loop over segments (hundreds) is
  fine, a Python loop over frames (200k) is not.

## 12. Data handling reminder

Never print raw tracking rows, tag instances or other data extracts into an LLM conversation —
aggregates only. See the `mtah-data` skill, §2.

## Example

Before:

```python
def tag_team(tagging, frame, window=None):
    if window is None:
        window = 200
    sp = tagging[...]
    if len(sp) == 0:
        return None
    return HOME if sp["team"].iloc[-1] == "H" else AWAY
```

After:

```python
def _team_of_latest_set_piece_before(
    tagging: pd.DataFrame, video_frame: int, search_window_frames: int
) -> int:
    """NO_TEAM when no team-tagged set piece opened in the window. Throw-ins are tagged with no
    team, so this is common and the caller falls back to the previously held team."""
    in_window = tagging[(tagging["code"] == "set pieces") & tagging["team"].notna()
                        & tagging["start_video_frame"].between(video_frame - search_window_frames,
                                                               video_frame)]
    team_codes = in_window["team"].map(domain.TEAM_CODE_BY_LETTER).to_numpy(dtype=np.int8)
    return int(team_codes[-1]) if len(team_codes) else domain.NO_TEAM
```
