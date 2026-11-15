"""Ball direction/speed-change ('kink') features: realism check on the real match, then LOGO vs 012."""
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, ".")
import games as challenge
from mtah import config, dead_ball, decode_viterbi, features, load, pipeline
from validation import labelled_games, degrade, synthetic_tagging, scoring
np.seterr(all="ignore")
c = config.DEFAULT_CONFIG


def rolling_max(x, w):
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(np.nan_to_num(x, nan=0.0), (w, w))
    return sliding_window_view(pad, 2 * w + 1).max(axis=1)


def kink_features(tracking, frame_features):
    L, W = tracking.pitch_length_m, tracking.pitch_width_m
    bx = tracking.ball_norm[:, 0].astype(float) * L; by = tracking.ball_norm[:, 1].astype(float) * W
    n = len(bx); cols = {}
    near_h = np.nan_to_num(frame_features.nearest_home_distance_m, nan=np.inf)
    near_a = np.nan_to_num(frame_features.nearest_away_distance_m, nan=np.inf)
    home_near = (near_h <= 2.0) & (near_h + 0.5 <= near_a); away_near = (near_a <= 2.0) & (near_a + 0.5 <= near_h)
    for w in (5, 10):
        vbx = np.full(n, np.nan); vby = np.full(n, np.nan); vax = np.full(n, np.nan); vay = np.full(n, np.nan)
        vbx[w:] = (bx[w:] - bx[:-w]) * 25 / w; vby[w:] = (by[w:] - by[:-w]) * 25 / w
        vax[:-w] = (bx[w:] - bx[:-w]) * 25 / w; vay[:-w] = (by[w:] - by[:-w]) * 25 / w
        kink = np.hypot(vax - vbx, vay - vby)
        sb, sa = np.hypot(vbx, vby), np.hypot(vax, vay)
        cosang = (vbx * vax + vby * vay) / np.maximum(sb * sa, 1e-6)
        signed = np.where(home_near, kink, np.where(away_near, -kink, 0.0))
        cols[f"kink_{w}"] = kink; cols[f"turn_cos_{w}"] = cosang; cols[f"speed_change_{w}"] = sa - sb
        cols[f"signed_kink_{w}"] = signed
        for r in (12, 25):
            cols[f"home_kink_max_{w}_{r}"] = rolling_max(np.where(home_near, kink, 0.0), r)
            cols[f"away_kink_max_{w}_{r}"] = rolling_max(np.where(away_near, kink, 0.0), r)
    return np.column_stack(list(cols.values())).astype(np.float32), list(cols)


# 1. Realism: kink distribution on the real match vs degraded validation games (no labels).
real = load.load_tracking(challenge.GAME, challenge.DATA_DIR, challenge.DATA_DIR + "/cache")
halves = load.load_graded_halves(challenge.GAME)
K, names = kink_features(real, features.compute_frame_features(real, c))
k = K[:, names.index("kink_5")]; g = np.zeros(len(k), bool)
for h in halves: g[h.first_video_frame:h.last_video_frame + 1] = True
k = k[g & np.isfinite(k)]
print(f"REAL match kink (0.2 s windows, m/s): p50 {np.percentile(k,50):.2f} p90 {np.percentile(k,90):.2f} p99 {np.percentile(k,99):.2f} | share > 5 m/s {np.mean(k>5):.1%}", flush=True)
vgames = labelled_games.load_all_labelled_games("data/external/metrica-sample-data", "data/external/idsse", "data/cache/validation")
for gm in (vgames[0], vgames[3]):
    t = degrade.degrade_labelled_game(gm, degrade.DEFAULT_DEGRADE_CONFIG, 0)
    Kg, _ = kink_features(t, features.compute_frame_features(t, c)); kk = Kg[:, names.index("kink_5")]; kk = kk[np.isfinite(kk)]
    print(f"{gm.name:12s} degraded kink: p50 {np.percentile(kk,50):.2f} p90 {np.percentile(kk,90):.2f} p99 {np.percentile(kk,99):.2f} | share > 5 m/s {np.mean(kk>5):.1%}", flush=True)

# 2. LOGO with kink features added.
rows = {}
for gm in vgames:
    t = degrade.degrade_labelled_game(gm, degrade.DEFAULT_DEGRADE_CONFIG, 0); f = features.compute_frame_features(t, c)
    X = np.hstack([pipeline.compute_model_inputs(t, f, gm.graded_halves, c), kink_features(t, f)[0]])
    live = np.flatnonzero((gm.truth == "H") | (gm.truth == "A"))[::c.model_training_frame_step]
    rows[gm.name] = (X[live], gm.truth[live] == "H")
n_base = len(pipeline.compute_model_inputs.__defaults__ or ()) or None
base_cols = None
per_game = {"012": {}, "+ ball kinks": {}}; pooled = {k_: ([], []) for k_ in per_game}
for gm in vgames:
    others = [rows[nm] for nm in rows if nm != gm.name]
    Xtr = np.vstack([o[0] for o in others]); ytr = np.concatenate([o[1] for o in others]).astype(np.int8)
    nb = Xtr.shape[1] - len(names)
    models = {}
    for label, cols in (("012", slice(0, nb)), ("+ ball kinks", slice(None))):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
        m.fit(Xtr[:, cols], ytr); models[label] = (m, cols)
    for label in per_game: per_game[label][gm.name] = ([], [])
    for seed in (0, 1, 2):
        t = degrade.degrade_labelled_game(gm, degrade.DEFAULT_DEGRADE_CONFIG, seed)
        tags = synthetic_tagging.synthesize_restart_tags(gm.truth, gm.graded_halves, seed)
        f = features.compute_frame_features(t, c); is_dead = dead_ball.detect_dead_ball(f, tags, gm.graded_halves, c)
        X = np.hstack([pipeline.compute_model_inputs(t, f, gm.graded_halves, c), kink_features(t, f)[0]])
        for label, (m, cols) in models.items():
            lab = decode_viterbi.decode_possession_labels(m.predict_proba(X[:, cols])[:, 1], is_dead, gm.graded_halves, c)
            per_game[label][gm.name][0].append(gm.truth); per_game[label][gm.name][1].append(lab)
            pooled[label][0].append(gm.truth); pooled[label][1].append(lab)
    print(gm.name, "done", flush=True)
score = lambda tp: scoring.score_labels(scoring.concatenate_label_runs(tp[0]), scoring.concatenate_label_runs(tp[1]), 0)
base = {g_: score(tp)["spell_f1"] for g_, tp in per_game["012"].items()}
for label in per_game:
    s = score(pooled[label]); gs = {g_: score(tp)["spell_f1"] for g_, tp in per_game[label].items()}
    d = np.array([gs[g_] - base[g_] for g_ in base])
    print(f"{label:14s} spellF1 {s['spell_f1']:.3f}  poss {s['predicted_possessions']:.0f}/{s['true_possessions']:.0f}  games better {np.sum(d>0)}/10  mean per-game {d.mean():+.3f}")
