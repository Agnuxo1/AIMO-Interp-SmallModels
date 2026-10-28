"""Animated figures (GIF) for the README: layer sweep, PCA separation by depth,
permutation null, and pipeline schematic. Inputs come from experiments/*.csv and features/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import cv_probe_lib as lib

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
BG, FG, MUTED = "#0f1420", "#e8ecf3", "#8a94a8"
ROBUST, SPUR = "#3ddc97", "#ff6b6b"
SERIES = ["#4ea8ff", "#ffb547", "#c792ea", "#3ddc97"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUTED, "axes.labelcolor": FG,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": FG, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
SHORT = {"Qwen/Qwen3.5-4B": "Qwen3.5-4B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "DeepSeek-R1-Qwen3-8B",
         "allenai/Olmo-3-7B-Think": "OLMo-3-7B-Think", "Skywork/Skywork-OR1-Math-7B": "Skywork-OR1-Math-7B"}


def layer_sweep_gif(sweep_csv="experiments/layer_sweep.csv"):
    d = pd.read_csv(sweep_csv)
    ens = pd.read_csv("experiments/ensembles.csv")
    ctrl = {"surface statistics": 0.749, "TF-IDF": 0.731}
    sel = ens[(ens.kinds == "last:band+mean:0.75+mean:1") & (ens.C == 0.1)].auc.iloc[0]
    depths = sorted(d.depth.unique())
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=100)

    def draw(k, final=False):
        ax.clear()
        for (name, v), ls in zip(ctrl.items(), ["--", ":"]):
            ax.axhline(v, color=MUTED, ls=ls, lw=1.2)
            ax.text(1.0, v + 0.004, f"{name} control {v:.3f}", color=MUTED, fontsize=8.5, ha="right")
        ax.axhline(sel, color=ROBUST, lw=1.4, alpha=0.9)
        ax.text(0.0, sel + 0.004, f"submitted ensemble {sel:.3f}", color=ROBUST, fontsize=9)
        for c, pool, label in [(SERIES[0], "last", "last prompt token"), (SERIES[1], "mean", "token mean")]:
            g = d[d.pool == pool].sort_values("depth")
            g = g[g.depth <= depths[k]]
            ax.fill_between(g.depth, g.auc - g.auc_sd, g.auc + g.auc_sd, color=c, alpha=0.18, lw=0)
            ax.plot(g.depth, g.auc, color=c, lw=2.4, label=label)
            if len(g):
                ax.scatter(g.depth.iloc[-1:], g.auc.iloc[-1:], color=c, s=42, zorder=3)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.45, 0.88)
        ax.set_xlabel("relative layer depth (0 = embeddings, 1 = final layer)")
        ax.set_ylabel("leave-problem-out AUROC")
        ax.set_title("Robustness becomes linearly decodable with depth", loc="left", fontsize=13)
        ax.legend(frameon=False, loc="lower right", fontsize=9)
    frames = list(range(len(depths))) + [len(depths) - 1] * 8
    anim = FuncAnimation(fig, draw, frames=frames, interval=180)
    anim.save(OUT / "layer_sweep.gif", writer=PillowWriter(fps=6))
    draw(len(depths) - 1)
    fig.savefig(OUT / "layer_sweep.png", dpi=150, facecolor=BG)
    plt.close(fig)


def pca_depth_gif(model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", pool="last", folds=5, seed=0):
    """Out-of-fold probe scores per layer (no leakage): separation of robust vs spurious grows with depth."""
    R = lib.rows_for(model, True).reset_index(drop=True)
    f = lib.FEATS[model]
    X = f[pool][[f["index"][p] for p in R.problem_id]]
    y = R.label.to_numpy().astype(int)
    g = R.problem_id.to_numpy()
    L = X.shape[1] - 1
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(g))
    fold = np.array([{u: i % folds for i, u in enumerate(uniq)}[x] for x in g])
    jitter = rng.uniform(-0.35, 0.35, len(y))
    layers = list(range(0, L + 1, max(1, L // 18)))
    scores = {}
    for l in layers:
        s_ = np.full(len(y), 0.5)
        for k in range(folds):
            tr, te = fold != k, fold == k
            if np.allclose(X[tr, l].std(0).max(), 0):
                continue
            Z = StandardScaler().fit(X[tr, l])
            p = PCA(n_components=16, random_state=seed).fit(Z.transform(X[tr, l]))
            lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(p.transform(Z.transform(X[tr, l])), y[tr])
            s_[te] = lr.predict_proba(p.transform(Z.transform(X[te, l])))[:, 1]
        scores[l] = s_
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=100)

    def draw(i):
        l = layers[i]
        ax.clear()
        s_ = scores[l]
        for lab_, c, name, y0 in [(1, ROBUST, "robust", 1), (0, SPUR, "spurious", 0)]:
            m = y == lab_
            ax.scatter(s_[m], y0 + jitter[m] * 0.6, s=16, color=c, alpha=0.75, label=name)
        ax.axvline(0.5, color=MUTED, ls="--", lw=1)
        auc = roc_auc_score(y, s_) if np.ptp(s_) > 0 else 0.5
        ax.set_xlim(0, 1); ax.set_ylim(-0.6, 1.6)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["spurious", "robust"])
        ax.set_xlabel("out-of-fold probe P(robust)")
        ax.set_title(f"{SHORT[model]} - layer {l}/{L} - AUROC {auc:.2f}", loc="left", fontsize=12)
    frames = list(range(len(layers))) + [len(layers) - 1] * 6
    anim = FuncAnimation(fig, draw, frames=frames, interval=260)
    anim.save(OUT / "probe_scores_by_depth.gif", writer=PillowWriter(fps=4))
    plt.close(fig)


def permutation_gif(perm_csv="experiments/permutation_null.csv", observed_json="experiments/permutation_observed.json"):
    null = pd.read_csv(perm_csv).auc.to_numpy()
    obs = json.load(open(observed_json))["auc"]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
    steps = np.linspace(10, len(null), 20).astype(int)

    def draw(k):
        ax.clear()
        ax.hist(null[:steps[k]], bins=np.linspace(0.2, 1.0, 41), color=MUTED, alpha=0.8, label="probe on shuffled labels")
        ax.axvline(obs, color=ROBUST, lw=3, label=f"real probe (AUROC {obs:.2f})")
        p = (1 + (null[:steps[k]] >= obs).sum()) / (1 + steps[k])
        ax.set_title(f"Permutation control: {steps[k]} shuffles, p = {p:.3f}", loc="left", fontsize=13)
        ax.set_xlabel("leave-problem-out AUROC"); ax.set_ylabel("count")
        ax.legend(frameon=False, loc="upper left")
    anim = FuncAnimation(fig, draw, frames=len(steps), interval=250)
    anim.save(OUT / "permutation_null.gif", writer=PillowWriter(fps=4))
    plt.close(fig)


def pipeline_gif():
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=100)
    boxes = [("Olympiad\nproblem", 0.07, 0.055), ("Evaluated LLM\n(one forward pass)", 0.27, 0.09),
             ("Residual stream\n(prompt only)", 0.50, 0.085), ("Linear probes\n(PCA + logistic)", 0.72, 0.085),
             ("Robust /\nSpurious", 0.92, 0.05)]

    def draw(k):
        ax.clear(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        for i, (t, x, hw) in enumerate(boxes):
            on = i <= k // 4
            ax.text(x, 0.5, t, ha="center", va="center", fontsize=11, color=FG if on else MUTED,
                    bbox=dict(boxstyle="round,pad=0.6", fc="#1b2333" if on else BG, ec=SERIES[i % 4] if on else MUTED, lw=2))
            if i < len(boxes) - 1 and i < k // 4:
                ax.annotate("", xy=(boxes[i + 1][1] - boxes[i + 1][2] - 0.012, 0.5), xytext=(x + hw + 0.012, 0.5),
                            arrowprops=dict(arrowstyle="-|>", color=FG, lw=1.6))
        if k // 4 >= 2:
            for j in range(12):
                h = 0.12 + 0.06 * np.sin(j + k)
                ax.add_patch(plt.Rectangle((0.47 + j * 0.01, 0.15), 0.007, h, color=SERIES[(j + k) % 4], alpha=0.8))
        ax.text(0.5, 0.92, "No generation at test time: one forward pass, ~0.05-0.3 s per problem on an RTX 3090", ha="center", color=MUTED, fontsize=10)
    anim = FuncAnimation(fig, draw, frames=24, interval=150)
    anim.save(OUT / "pipeline.gif", writer=PillowWriter(fps=6))
    plt.close(fig)


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:] or ["pipeline", "layer_sweep", "probe_scores", "permutation"]:
        {"pipeline": pipeline_gif, "layer_sweep": layer_sweep_gif, "probe_scores": pca_depth_gif,
         "permutation": permutation_gif}[name]()
        print("wrote", name, flush=True)
