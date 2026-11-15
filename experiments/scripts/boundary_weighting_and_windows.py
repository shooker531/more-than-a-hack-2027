"""(1) Switch-error breakdown for 012. (2) Boundary-weighted training and shorter windows, per game."""
import sys, collections
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, ".")
from mtah import config, dead_ball, decode_viterbi, features, model_features, pipeline
from validation import labelled_games, degrade, synthetic_tagging, scoring
np.seterr(all="ignore")
c = config.DEFAULT_CONFIG
NAMES = model_features.FEATURE_NAMES
ALL_COLS = list(range(len(NAMES)))
NO_LONG_WINDOW = [i for i, n in enumerate(NAMES) if not n.endswith("_mean_125f")]


def team_switches(labels):
    team = np.where(labels == "H", 0, np.where(labels == "A", 1, -1))
    idx = np.flatnonzero((team[1:] >= 0) & (team[:-1] >= 0) & (team[1:] != team[:-1])) + 1
    return idx, team[idx]


def near_switch_weights(truth, weight):
    w = np.ones(len(truth))
    for s in team_switches(truth)[0]:
        w[max(0, s - 25):s + 26] = weight
    return w


def train(rows, cols, weighted):
    X = np.vstack([r[0][:, cols] for r in rows]); y = np.concatenate([r[1] for r in rows])
    w = np.concatenate([r[2] for r in rows]) if weighted else None
    m = HistGradientBoostingClassifier(max_iter=c.model_max_iterations, learning_rate=c.model_learning_rate,
                                       max_leaf_nodes=c.model_max_leaf_nodes, l2_regularization=c.model_l2_regularisation,
                                       random_state=c.model_random_seed)
    m.fit(X, y.astype(np.int8), sample_weight=w)
    return m


games = labelled_games.load_all_labelled_games("data/external/metrica-sample-data", "data/external/idsse", "data/cache/validation")
rows = {}
for g in games:
    t = degrade.degrade_labelled_game(g, degrade.DEFAULT_DEGRADE_CONFIG, 0)
    X = pipeline.compute_model_inputs(t, features.compute_frame_features(t, c), g.graded_halves, c)
    live = np.flatnonzero((g.truth == "H") | (g.truth == "A"))[::c.model_training_frame_step]
    rows[g.name] = (X[live], g.truth[live] == "H", near_switch_weights(g.truth, 3.0)[live])

variants = {"012": (ALL_COLS, False), "A: weight switches x3": (ALL_COLS, True),
            "B: drop +-2.5 s window": (NO_LONG_WINDOW, False), "A+B": (NO_LONG_WINDOW, True)}
per_game = {v: {} for v in variants}
pooled = {v: ([], []) for v in variants}
breakdown = collections.Counter(); n_true = 0; false_switches = 0; n_pred = 0; wrong_long = 0; live_total = 0
for g in games:
    others = [rows[n] for n in rows if n != g.name]
    models = {v: train(others, cols, weighted) for v, (cols, weighted) in variants.items()}
    for v in variants: per_game[v][g.name] = ([], [])
    for seed in (0, 1, 2):
        t = degrade.degrade_labelled_game(g, degrade.DEFAULT_DEGRADE_CONFIG, seed)
        tags = synthetic_tagging.synthesize_restart_tags(g.truth, g.graded_halves, seed)
        f = features.compute_frame_features(t, c)
        is_dead = dead_ball.detect_dead_ball(f, tags, g.graded_halves, c)
        X = pipeline.compute_model_inputs(t, f, g.graded_halves, c)
        for v, (cols, _) in variants.items():
            labels = decode_viterbi.decode_possession_labels(models[v].predict_proba(X[:, cols])[:, 1], is_dead, g.graded_halves, c)
            per_game[v][g.name][0].append(g.truth); per_game[v][g.name][1].append(labels)
            pooled[v][0].append(g.truth); pooled[v][1].append(labels)
            if v == "012":
                ts, tt = team_switches(g.truth); ps, pt = team_switches(labels)
                for s, team in zip(ts, tt):
                    n_true += 1
                    same = ps[pt == team]
                    d = np.min(np.abs(same - s)) / 25 if len(same) else np.inf
                    breakdown["within 0.5 s" if d <= .5 else "0.5-1 s off" if d <= 1 else "1-2 s off" if d <= 2 else "2-5 s off" if d <= 5 else "missed (none within 5 s)"] += 1
                for s, team in zip(ps, pt):
                    n_pred += 1
                    same = ts[tt == team]
                    false_switches += int(not len(same) or np.min(np.abs(same - s)) > 125)
                live = (g.truth == "H") | (g.truth == "A")
                wrong = live & ((labels == "H") | (labels == "A")) & (labels != g.truth)
                live_total += live.sum()
                wrong_long += sum(b - a for a, b in __import__("mtah").runs.true_runs(wrong) if b - a >= 125)
    print(g.name, "done", flush=True)

print(f"\n012 SWITCH BREAKDOWN ({n_true} true open-play turnovers; nearest predicted switch to the same team):")
for k in ("within 0.5 s", "0.5-1 s off", "1-2 s off", "2-5 s off", "missed (none within 5 s)"):
    print(f"  {k:26s} {breakdown[k]/n_true:5.0%}")
print(f"  predicted switches with no true switch within 5 s: {false_switches}/{n_pred} ({false_switches/n_pred:.0%})")
print(f"  live frames inside wrong-team stretches lasting >= 5 s: {wrong_long/live_total:.1%} of live frames")

print("\nVARIANTS (pooled, and per game vs 012):")
for v in variants:
    s = scoring.score_labels(scoring.concatenate_label_runs(pooled[v][0]), scoring.concatenate_label_runs(pooled[v][1]), 0)
    game_scores = {gname: scoring.score_labels(scoring.concatenate_label_runs(tp[0]), scoring.concatenate_label_runs(tp[1]), 0)["spell_f1"] for gname, tp in per_game[v].items()}
    base = {gname: scoring.score_labels(scoring.concatenate_label_runs(tp[0]), scoring.concatenate_label_runs(tp[1]), 0)["spell_f1"] for gname, tp in per_game["012"].items()}
    d = np.array([game_scores[k] - base[k] for k in base])
    print(f"  {v:24s} spellF1 {s['spell_f1']:.3f}  poss {s['predicted_possessions']:.0f}/{s['true_possessions']:.0f}  games better {np.sum(d > 0)}/10  mean per-game {d.mean():+.3f}")
