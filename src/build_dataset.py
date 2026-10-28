"""Collect every public AIMO-Interp problem and derive max-drop labels.

Writes data/problems.jsonl (unique problem texts) and data/labels.csv
(one row per problem x labelling model x source).
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT.parent / "data"
BASE = ROOT.parent / "baselines" / "data"


def pid(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:12]


def label(drop: float):
    if drop <= 0.10:
        return 1
    if drop >= 0.25:
        return 0
    return None


rows = []
train = pd.read_parquet(RAW / "train-main-v2.parquet")
for r in train.itertuples():
    rows.append(dict(problem=r.problem, model=f"{r.model_id}|{r.reasoning_effort}", source="train-main-v2",
                     max_drop=float(r.max_drop), official_label=None if pd.isna(r.is_robust) else int(r.is_robust)))

# Per-perturbation-family rows: max over families of the absolute accuracy decay.
fin = pd.read_csv(BASE / "math-robust-final.csv")
for (prob, model), g in fin.groupby(["original_problem", "model_id"]):
    rows.append(dict(problem=prob, model=model, source="baselines-math-final",
                     max_drop=float(g.absolute_accuracy_decay.max()), official_label=None))

for name in ["augmented-sample-math-agg-filtered", "val-sample"]:
    d = pd.read_parquet(RAW / f"{name}.parquet")
    for r in d.itertuples():
        rows.append(dict(problem=r.original_problem, model=r.model_id, source=name,
                         max_drop=float(r.absolute_accuracy_decay), official_label=int(r.model_is_robust)))

full = pd.read_parquet(RAW / "aimo-interp-challenge-sample-full.parquet")
for (prob, model), g in full.groupby(["original_problem", "model_id"]):
    rows.append(dict(problem=prob, model=model, source="challenge-sample-full",
                     max_drop=float(g.absolute_accuracy_decay.max()), official_label=None))

lab = pd.DataFrame(rows)
(ROOT / 'data').mkdir(parents=True, exist_ok=True)
lab["problem_id"] = lab.problem.map(pid)
lab["label"] = lab.max_drop.map(label)
lab.drop(columns="problem").to_csv(ROOT / "data" / "labels.csv", index=False)
probs = lab.drop_duplicates("problem_id")[["problem_id", "problem"]]
with open(ROOT / "data" / "problems.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for r in probs.itertuples():
        f.write(json.dumps({"id": r.problem_id, "problem": r.problem}, ensure_ascii=False) + "\n")
print(len(probs), "problems;", len(lab), "label rows")
print(lab.groupby("source").agg(n=("label", "size"), robust=("label", lambda s: (s == 1).sum()),
                                spurious=("label", lambda s: (s == 0).sum())))
chk = lab[lab.official_label.notna() & lab.label.notna()]
print("agreement derived vs official label:", (chk.label == chk.official_label).mean(), len(chk))
