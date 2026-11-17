"""Oracle decomposition: how much spellF1 headroom is in dead-ball vs team switching?"""
import sys
import numpy as np
sys.path.insert(0, ".")
from mtah import config, dead_ball, decode_viterbi, domain, features, pipeline, possession_model
from validation import labelled_games, degrade, synthetic_tagging, training_data, scoring
np.seterr(all="ignore")
c = config.DEFAULT_CONFIG
games = labelled_games.load_all_labelled_games("data/external/metrica-sample-data", "data/external/idsse", "data/cache/validation")
examples = {g.name: training_data.build_training_examples(g, c) for g in games}


def truth_team_filled(truth):
    """Truth team per frame, carried through dead stretches (last team before, else next)."""
    team = np.where(truth == "H", 0, np.where(truth == "A", 1, -1))
    idx = np.where(team >= 0, np.arange(len(team)), -1)
    forward = np.maximum.accumulate(idx)
    filled = np.where(forward >= 0, team[np.maximum(forward, 0)], -1)
    first = np.flatnonzero(team >= 0)[0]
    filled[filled < 0] = team[first]
    return filled


variants = ["current pipeline", "oracle dead ball, predicted team", "predicted dead ball, oracle team", "oracle both (sanity)"]
pooled = {v: ([], []) for v in variants}
dead_stats = []
for g in games:
    tr = training_data.combine_training_examples([examples[n] for n in examples if n != g.name])
    model = possession_model.train_possession_model(tr.feature_rows, tr.is_home_in_possession, c)
    true_dead = g.truth == "D"
    team_filled = truth_team_filled(g.truth)
    graded = g.truth != "X"
    for seed in (0, 1, 2):
        t = degrade.degrade_labelled_game(g, degrade.DEFAULT_DEGRADE_CONFIG, seed)
        tags = synthetic_tagging.synthesize_restart_tags(g.truth, g.graded_halves, seed)
        f = features.compute_frame_features(t, c)
        pred_dead = dead_ball.detect_dead_ball(f, tags, g.graded_halves, c)
        p_home = possession_model.predict_home_probability(model, pipeline.compute_model_inputs(t, f, g.graded_halves, c))
        current = decode_viterbi.decode_possession_labels(p_home, pred_dead, g.graded_halves, c)
        oracle_dead = decode_viterbi.decode_possession_labels(p_home, true_dead, g.graded_halves, c)
        oracle_team = np.where(~graded, "X", np.where(pred_dead, "D", np.where(team_filled == 0, "H", "A")))
        for v, labels in zip(variants, (current, oracle_dead, oracle_team, g.truth)):
            pooled[v][0].append(g.truth); pooled[v][1].append(labels)
        dead_stats.append((np.sum(graded & pred_dead & ~true_dead), np.sum(graded & ~pred_dead & true_dead), np.sum(graded)))
    print(g.name, "done", flush=True)

print()
for v in variants:
    s = scoring.score_labels(scoring.concatenate_label_runs(pooled[v][0]), scoring.concatenate_label_runs(pooled[v][1]), 300)
    print(f"{v:36s} spellF1 {s['spell_f1']:.3f} [{s['spell_f1_low']:.3f},{s['spell_f1_high']:.3f}]  macro {s['macro_f1']:.3f}  poss {s['predicted_possessions']:.0f}/{s['true_possessions']:.0f}")
d = np.array(dead_stats).sum(axis=0)
print(f"\ndead-ball errors: live called dead {d[0]/d[2]:.1%} of graded frames, dead called live {d[1]/d[2]:.1%}")
