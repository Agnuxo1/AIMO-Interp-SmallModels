"""E2-E5: ensembles, layer sweep, permutation control and paired bootstrap CIs.

Same grouped-by-problem protocol as cv_probe.py; out-of-fold predictions are
kept so that paired comparisons are made on identical rows.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cv_probe_lib import SMALL, featurize, probs, rows_for

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=["ensembles", "sweep", "perm", "bootstrap", "weights"])
ap.add_argument("--kind", default="last:band")
ap.add_argument("--C", type=float, default=0.01)
ap.add_argument("--perms", type=int, default=200)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--eval-all", action="store_true", help="evaluate on own AND proxy (problem-level) labels")
ap.add_argument("--own-weight", type=float, default=1.0)
args = ap.parse_args()
OUT = Path("experiments")
OUT.mkdir(exist_ok=True)


def clf_for(kind, C, seed, n_train):
    if kind == "tfidf":
        return make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2)),
                             LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed))
    steps = [StandardScaler()]
    if kind != "length":
        steps.append(PCA(n_components=min(16, n_train - 1), random_state=seed))
    steps.append(LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed))
    return make_pipeline(*steps)


def oof(kinds, C, seed, shuffle=False):
    """Out-of-fold probabilities on own-model rows; ensemble = mean probability over kinds."""
    rng = np.random.default_rng(seed)
    recs = []
    for target in SMALL:
        R = rows_for(target, True).reset_index(drop=True)
        y = R.label.to_numpy().astype(int)
        if shuffle:
            y = rng.permutation(y)
        g = R.problem_id.to_numpy()
        uniq = rng.permutation(np.unique(g))
        fold = np.array([{u: i % args.folds for i, u in enumerate(uniq)}[x] for x in g])
        Xs = {k: ([probs[p] for p in g] if k == "tfidf" else
                  None if k.endswith(':band') else featurize(target, k, g)) for k in kinds}
        for f in range(args.folds):
            te = (fold == f) & ((R.own.to_numpy() == 1) | args.eval_all)
            tr = fold != f
            if te.sum() == 0 or len(set(y[tr])) < 2:
                continue
            ps = []
            for k in kinds:
                X = (featurize(target, k, g, train_mask=tr)
                     if k.endswith(':band') else Xs[k])
                pick = (lambda m: [X[i] for i in np.where(m)[0]]) if k == "tfidf" else (lambda m: X[m])
                c = clf_for(k, C, seed, int(tr.sum()))
                w = np.where(R.own.to_numpy()[tr] == 1, args.own_weight, 1.0)
                c.fit(pick(tr), y[tr], logisticregression__sample_weight=w)
                ps.append(c.predict_proba(pick(te))[:, 1])
            p = np.mean(ps, axis=0)
            for i, pi in zip(np.where(te)[0], p):
                recs.append(dict(model=target, problem_id=g[i], source=R.source[i], y=y[i], p=pi, seed=seed))
    return pd.DataFrame(recs)


def metrics(d):
    yp = (d.p >= 0.5).astype(int)
    o = (d.source == "train-main-v2").to_numpy()
    if o.sum() == 0 or d.y[o].nunique() < 2:
        o = np.ones(len(d), bool)
    return dict(acc=(yp == d.y).mean(), bacc=balanced_accuracy_score(d.y, yp), auc=roc_auc_score(d.y, d.p),
                oly_bacc=balanced_accuracy_score(d.y[o], yp[o]), oly_auc=roc_auc_score(d.y[o], d.p[o]))


def summarize(kinds, C):
    ms = [metrics(oof(kinds, C, s)) for s in range(args.seeds)]
    return {**{k: float(np.mean([m[k] for m in ms])) for k in ms[0]},
            **{k + "_sd": float(np.std([m[k] for m in ms])) for k in ms[0]}}


if args.task == "ensembles":
    cands = [["last:band"], ["mean:1"], ["last:0.5"], ["last:band", "mean:1"], ["last:band", "mean:1", "last:0.5"],
             ["last:band", "mean:0.75", "mean:1"], ["last:band", "length"], ["last:band", "mean:1", "length"],
             ["length"], ["tfidf"], ["mean:0"]]
    rows = []
    for kinds in cands:
        for C in [0.01, 0.1]:
            r = dict(kinds="+".join(kinds), C=C, **summarize(kinds, C))
            rows.append(r)
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "ensembles.csv", index=False)

elif args.task == "weights":
    rows = []
    for w in [1.0, 2.0, 4.0]:
        args.own_weight = w
        r = dict(own_weight=w, kinds=args.kind, C=args.C, **summarize(args.kind.split("+"), args.C))
        rows.append(r)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "own_weight.csv", index=False)

elif args.task == "sweep":
    rows = []
    for pool in ["last", "mean"]:
        for depth in np.round(np.linspace(0, 1, 21), 2):
            r = dict(pool=pool, depth=float(depth), **summarize([f"{pool}:{depth}"], args.C))
            rows.append(r)
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "layer_sweep.csv", index=False)

elif args.task == "perm":
    kinds = args.kind.split("+")
    obs = summarize(kinds, args.C)
    null = []
    for i in range(args.perms):
        null.append(metrics(oof(kinds, args.C, 1000 + i, shuffle=True)))
    nd = pd.DataFrame(null)
    nd.to_csv(OUT / "permutation_null.csv", index=False)
    p = {k: float((1 + (nd[k] >= obs[k]).sum()) / (1 + len(nd))) for k in ["auc", "bacc", "oly_auc"]}
    json.dump(dict(kinds=args.kind, C=args.C, **obs, p_values=p, perms=args.perms),
              open(OUT / "permutation_observed.json", "w"), indent=1)
    print(json.dumps(dict(obs=obs, p=p)))

elif args.task == "bootstrap":
    # Paired bootstrap over problems: selected probe vs each control, seed-averaged OOF probabilities.
    def avg_oof(kinds):
        d = pd.concat([oof(kinds, args.C, s) for s in range(args.seeds)])
        return d.groupby(["model", "problem_id", "source", "y"], as_index=False).p.mean()
    sel = avg_oof(args.kind.split("+"))
    out = {}
    rng = np.random.default_rng(0)
    for ctrl in [["length"], ["tfidf"], ["mean:0"]]:
        c = avg_oof(ctrl)
        m = sel.merge(c, on=["model", "problem_id", "source", "y"], suffixes=("_s", "_c"))
        pids = m.problem_id.unique()
        rows_of = {q: np.where(m.problem_id.to_numpy() == q)[0] for q in pids}
        Y, PS, PC = m.y.to_numpy(), m.p_s.to_numpy(), m.p_c.to_numpy()
        diffs = {"auc": [], "bacc": []}
        for _ in range(2000):
            ix = np.concatenate([rows_of[q] for q in rng.choice(pids, size=len(pids), replace=True)])
            y_, s_, c_ = Y[ix], PS[ix], PC[ix]
            if len(set(y_)) < 2:
                continue
            diffs["auc"].append(roc_auc_score(y_, s_) - roc_auc_score(y_, c_))
            diffs["bacc"].append(balanced_accuracy_score(y_, s_ >= .5) - balanced_accuracy_score(y_, c_ >= .5))
        point = dict(auc=roc_auc_score(m.y, m.p_s) - roc_auc_score(m.y, m.p_c),
                     bacc=balanced_accuracy_score(m.y, m.p_s >= .5) - balanced_accuracy_score(m.y, m.p_c >= .5))
        out["+".join(ctrl)] = {k: dict(delta=float(point[k]), ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])
                               for k, v in diffs.items()}
    sel.to_csv(OUT / ("oof_selected_all.csv" if args.eval_all else "oof_selected.csv"), index=False)
    json.dump(dict(selected=args.kind, C=args.C, eval_all=args.eval_all, n_rows=int(len(sel)), vs=out), open(OUT / ("bootstrap_ci_all.json" if args.eval_all else "bootstrap_ci.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
