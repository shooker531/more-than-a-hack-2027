"""Learned switch model -> per-frame switch costs in Viterbi. Leave-one-game-out vs 012."""
import sys, itertools
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, ".")
from mtah import config, dead_ball, decode_viterbi, features, model_features, pipeline, possession_model, runs
from validation import labelled_games, degrade, synthetic_tagging, scoring
np.seterr(all="ignore")
c = config.DEFAULT_CONFIG
NAMES = model_features.FEATURE_NAMES
CONTRAST_SOURCES = [NAMES.index(n) for n in ("touch_evidence", "control_evidence", "nearest_difference_m",
                                              "advance_difference_m", "pressing_difference")]
WINDOWS = (12, 25, 50)
SWITCH_RADIUS = 12


def window_means(x, w):
    valid = ~np.isnan(x); cs = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0)))); cn = np.concatenate(([0], np.cumsum(valid)))
    i = np.arange(len(x))
    def mean(lo, hi):
        lo = np.clip(lo, 0, len(x)); hi = np.clip(hi, 0, len(x)); n = cn[hi] - cn[lo]
        return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
    return mean(i - w, i), mean(i, i + w)


def switch_features(X):
    cols = [X]
    for j in CONTRAST_SOURCES:
        for w in WINDOWS:
            before, after = window_means(X[:, j].astype(float), w)
            cols += [(after - before)[:, None], np.abs(after - before)[:, None]]
    return np.hstack(cols).astype(np.float32)


def switch_labels(truth):
    team = np.where(truth == "H", 0, np.where(truth == "A", 1, -1))
    sw = np.flatnonzero((team[1:] >= 0) & (team[:-1] >= 0) & (team[1:] != team[:-1])) + 1
    near = np.zeros(len(truth), bool)
    for s in sw: near[max(0, s - SWITCH_RADIUS):s + SWITCH_RADIUS + 1] = True
    return near


def viterbi_varying(logp, costs):
    n = len(logp); came = np.zeros((n, 2), bool); bh, ba = float(logp[0, 0]), float(logp[0, 1])
    h, a, cc = logp[:, 0].tolist(), logp[:, 1].tolist(), costs.tolist()
    for t in range(1, n):
        hf, af = ba - cc[t], bh - cc[t]
        came[t, 0], came[t, 1] = hf > bh, af > ba
        bh, ba = max(bh, hf) + h[t], max(ba, af) + a[t]
    s = np.zeros(n, np.int8); s[-1] = 0 if bh >= ba else 1
    for t in range(n - 1, 0, -1):
        s[t - 1] = 1 - s[t] if came[t, s[t]] else s[t]
    return s


def decode(p_home, p_switch, is_dead, halves, base, weight, prior):
    pf = np.clip(p_home, 1e-3, 1 - 1e-3); logp = np.column_stack([np.log(pf), np.log(1 - pf)])
    ps = np.clip(p_switch, 1e-4, 1 - 1e-4)
    logit = np.log(ps / (1 - ps)) - np.log(prior / (1 - prior))
    costs = np.maximum(1.0, base - weight * logit)
    labels = np.full(len(p_home), "X", dtype="<U1")
    for hh in halves:
        lo, hi = hh.first_video_frame, hh.last_video_frame + 1; labels[lo:hi] = "D"
        for a, b in runs.true_runs(~is_dead[lo:hi]):
            s = viterbi_varying(logp[lo + a:lo + b], costs[lo + a:lo + b])
            labels[lo + a:lo + b] = np.where(s == 0, "H", "A")
    return labels


games = labelled_games.load_all_labelled_games("data/external/metrica-sample-data", "data/external/idsse", "data/cache/validation")
train_rows = {}
for g in games:
    t = degrade.degrade_labelled_game(g, degrade.DEFAULT_DEGRADE_CONFIG, 0)
    X = pipeline.compute_model_inputs(t, features.compute_frame_features(t, c), g.graded_halves, c)
    live = np.flatnonzero((g.truth == "H") | (g.truth == "A"))
    idx = live[::c.model_training_frame_step]
    train_rows[g.name] = (X[idx], g.truth[idx] == "H", switch_features(X)[idx], switch_labels(g.truth)[idx])
print("switch label rate on live frames:", round(float(np.mean(np.concatenate([r[3] for r in train_rows.values()]))), 3), flush=True)

grid = [(20.0, 0.0)] + list(itertools.product((20.0, 30.0, 40.0), (3.0, 6.0, 10.0)))
pooled = {k: ([], []) for k in grid}
for g in games:
    others = [train_rows[n] for n in train_rows if n != g.name]
    pm = possession_model.train_possession_model(np.vstack([o[0] for o in others]), np.concatenate([o[1] for o in others]), c)
    sw_y = np.concatenate([o[3] for o in others]); prior = float(sw_y.mean())
    sm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    sm.fit(np.vstack([o[2] for o in others]), sw_y.astype(np.int8))
    for seed in (0, 1, 2):
        t = degrade.degrade_labelled_game(g, degrade.DEFAULT_DEGRADE_CONFIG, seed)
        tags = synthetic_tagging.synthesize_restart_tags(g.truth, g.graded_halves, seed)
        f = features.compute_frame_features(t, c)
        is_dead = dead_ball.detect_dead_ball(f, tags, g.graded_halves, c)
        X = pipeline.compute_model_inputs(t, f, g.graded_halves, c)
        p_home = possession_model.predict_home_probability(pm, X)
        p_switch = sm.predict_proba(switch_features(X))[:, 1]
        for base, weight in grid:
            pooled[(base, weight)][0].append(g.truth)
            pooled[(base, weight)][1].append(decode(p_home, p_switch, is_dead, g.graded_halves, base, weight, prior))
    print(g.name, "done", flush=True)

for (base, weight), (tr, pr) in pooled.items():
    s = scoring.score_labels(scoring.concatenate_label_runs(tr), scoring.concatenate_label_runs(pr), 0)
    tag = "fixed cost (012)" if weight == 0 else f"base {base:.0f}, weight {weight:.0f}"
    print(f"{tag:22s} spellF1 {s['spell_f1']:.3f}  macro {s['macro_f1']:.3f}  poss {s['predicted_possessions']:.0f}/{s['true_possessions']:.0f}", flush=True)
