"""Shared data loading for probe cross-validation and export."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from band_normalization import normalize_band

FEATURES_DIR = os.environ.get("AIMO_FEATURES", "features")

SMALL = {
    "Qwen/Qwen3.5-4B": "Qwen_Qwen3.5-4B",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B",
    "allenai/Olmo-3-7B-Think": "allenai_Olmo-3-7B-Think",
    "Skywork/Skywork-OR1-Math-7B": "Skywork_Skywork-OR1-Math-7B",
}

lab = pd.read_csv("data/labels.csv").dropna(subset=["label"])
lab["eval_model"] = lab.model.str.split("|").str[0]
probs = {json.loads(l)["id"]: json.loads(l)["problem"] for l in open("data/problems.jsonl", encoding="utf-8")}

FEATS = {}
for mid, safe in SMALL.items():
    z = np.load(Path(FEATURES_DIR) / f"{safe}.npz")
    FEATS[mid] = {"index": {p: i for i, p in enumerate(z["ids"])}, "last": z["last"].astype(np.float32),
                  "mean": z["mean"].astype(np.float32)}


def rows_for(target, pooled):
    own = lab[lab.eval_model == target]
    if not pooled:
        return own[["problem_id", "label", "source"]].assign(own=1)
    other = lab[lab.eval_model != target]
    # Problem-level proxy: majority label of other labelling models, only where the target has no own label.
    other = other[~other.problem_id.isin(own.problem_id)]
    agg = other.groupby("problem_id").label.mean().round().reset_index()
    return pd.concat([own[["problem_id", "label", "source"]].assign(own=1), agg.assign(own=0, source="proxy")], ignore_index=True)


def featurize(target, kind, pids, train_mask=None):
    f = FEATS[target]
    idx = [f["index"][p] for p in pids]
    if kind == "length":
        t = [probs[p] for p in pids]
        return np.array([[len(s), s.count("$"), s.count("\\"), len(s.split())] for s in t], dtype=np.float32)
    pool, layer = kind.split(":")
    X = f[pool][idx]
    L = X.shape[1] - 1
    if layer == "band":  # mean of standardised middle-to-late layers (0.4L..0.8L)
        lo, hi = int(0.4 * L), int(0.8 * L)
        B = X[:, lo:hi + 1]
        if train_mask is None:
            raise ValueError('Band CV requires a training-row mask; full-data normalization is transductive')
        return normalize_band(B, train_mask)
    return X[:, int(float(layer) * L)]  # relative depth
