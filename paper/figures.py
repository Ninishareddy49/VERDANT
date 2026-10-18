"""Publication figures for the VERDANT paper (IEEE two-column sizing).

    python paper/figures.py

Reads paper/results/*.json (produced by experiments.py / retrieval_eval.py) and the
training log stored in train_models.ipynb; writes PNG (600 dpi) + PDF to paper/figures/.
"""
from __future__ import annotations

import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

COL, DCOL = 3.45, 7.16  # IEEE single / double column width (in)
GREEN, TEAL, ORANGE, BLUE, RED, PURPLE, GREY, LIGHT = "#2E7D5B", "#2A9D8F", "#E07A1F", "#3366A8", "#C0392B", "#7B5EA7", "#6B6B6B", "#D9E7DF"
GREENS = LinearSegmentedColormap.from_list("vg", ["#F4F9F6", "#A8D5BA", "#2E7D5B", "#123D2B"])
DIVERGE = LinearSegmentedColormap.from_list("vd", ["#3366A8", "#F7F7F7", "#C0392B"])

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E3E3E3", "grid.linewidth": 0.5, "grid.linestyle": "-",
    "legend.frameon": False, "savefig.dpi": 600, "figure.dpi": 150, "lines.linewidth": 1.2,
})


def load(name):
    with open(os.path.join(C.RESULTS_DIR, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def sublabel(ax, text, y=None):
    """Register a sub-figure label; out() places all labels on one shared baseline."""
    fig = ax.figure
    if not hasattr(fig, "_sub"):
        fig._sub = []
    fig._sub.append((ax, text))


def _place_sublabels(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    axes = [a for a, _ in fig._sub]
    bottom = min(inv.transform((0, bb.y0))[1] for bb in (a.get_tightbbox(r) for a in axes))
    for ax, text in fig._sub:
        p = ax.get_position()
        fig.text((p.x0 + p.x1) / 2, bottom - 0.03, text, ha="center", va="top", fontsize=7.5)


def out(fig, name):
    if hasattr(fig, "_sub"):
        _place_sublabels(fig)
    fig.savefig(os.path.join(C.FIG_DIR, f"{name}.png"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(C.FIG_DIR, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  figure:", name)


def training_log():
    nb = json.load(open(os.path.join(C.ROOT, "train_models.ipynb"), encoding="utf-8"))
    text = "".join("".join(o.get("text", "")) for c in nb["cells"] for o in c.get("outputs", []))
    rows = re.findall(r"Epoch \[(\d+)/15\].*?Train Loss: ([\d.]+).*?Train Acc: ([\d.]+)%.*?Val Loss: ([\d.]+).*?Val Acc: ([\d.]+)%", text)
    return np.array(rows, dtype=float)


def heatmap(ax, M, labels_x, labels_y, cmap, fmt, vmin=None, vmax=None, annot=None):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(labels_x)), labels_x)
    ax.set_yticks(range(len(labels_y)), labels_y)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    lo, hi = (vmin if vmin is not None else M.min()), (vmax if vmax is not None else M.max())
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            light = (v - lo) / (hi - lo + 1e-9)
            txt = annot[i][j] if annot is not None else format(v, fmt)
            ax.text(j, i, txt, ha="center", va="center", fontsize=6, color="white" if (light > 0.6 or (cmap is DIVERGE and abs(v) > 0.55)) else "#222222")
    return im


# --------------------------------------------------------------------------- Fig: datasets
def fig_datasets(V):
    fig = plt.figure(figsize=(DCOL, 1.95))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1.15, 1, 1.25], wspace=0.42)

    # (a) leaf samples
    leaf = C.build_leaf_dataset()
    ax = fig.add_subplot(gs[0])
    healthy = [p for p, l in leaf.val_files if l == 0][:3]
    blight = [p for p, l in leaf.val_files if l == 1][:3]
    tiles = [np.hstack([np.array(Image.open(p)) for p in row]) for row in (healthy, blight)]
    pad = np.full((6, tiles[0].shape[1], 3), 255, dtype=np.uint8)
    ax.imshow(np.vstack([tiles[0], pad, tiles[1]]))
    ax.set_yticks([64, 198], ["Healthy", "Late\nblight"])
    ax.set_xticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    sublabel(ax, "(a) Leaf image samples", y=-0.12)

    # (b) correlation
    ax = fig.add_subplot(gs[1])
    corr = np.array(V["crop"]["corr"])
    names = ["N", "P", "K", "Temp", "Hum", "pH", "Rain"]
    heatmap(ax, corr, names, names, DIVERGE, ".2f", -1, 1, annot=[[f"{v:.1f}".replace("-0.0", "0.0").replace("1.0", "1") for v in r] for r in corr])
    ax.tick_params(axis="x", rotation=90)
    sublabel(ax, "(b) Crop feature correlation", y=-0.36)

    # (c) lettuce distribution
    ax = fig.add_subplot(gs[2])
    days = np.array(V["lettuce"]["all_days"])
    ax.hist(days, bins=18, color=GREEN, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(days.mean(), color=ORANGE, ls="--", lw=1)
    ax.text(0.98, 0.97, f"mean = {days.mean():.1f} d", transform=ax.transAxes, ha="right", va="top", color=ORANGE, fontsize=6)
    ax.set_xlabel("Days to harvest")
    ax.set_ylabel("Cycles")
    sublabel(ax, "(c) Lettuce growth cycles", y=-0.36)

    # (d) sensors
    ax = fig.add_subplot(gs[3])
    s = V["forecast"]["sensors"]
    idx = pd.to_datetime(s["index"])
    k = 3 * 144
    hours = (idx[:k] - idx[0]).total_seconds() / 3600
    ax.plot(hours, np.array(s["temperature"])[:k], color=RED, lw=0.7, label="Temp. (°C)")
    ax.set_ylabel("Temperature (°C)", color=RED)
    ax.tick_params(axis="y", colors=RED)
    ax2 = ax.twinx()
    ax2.plot(hours, np.array(s["co2"])[:k] / 10, color=BLUE, lw=0.7, alpha=0.8, label="CO$_2$ (×10 ppm)")
    ax2.plot(hours, np.array(s["humidity"])[:k], color=TEAL, lw=0.7, alpha=0.8, label="RH (%)")
    ax2.set_ylabel("RH (%) / CO$_2$ (×10 ppm)")
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax.set_xticks([0, 24, 48, 72])
    ax.set_xlabel("Time (h)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, handlelength=1.2, columnspacing=0.8)
    sublabel(ax, "(d) Greenhouse sensor streams", y=-0.36)
    out(fig, "fig_datasets")


# --------------------------------------------------------------------------- Fig: CNN
def fig_cnn(V, A, B):
    log = training_log()
    ep = log[:, 0]
    fig, axs = plt.subplots(1, 4, figsize=(DCOL, 1.75), gridspec_kw={"wspace": 0.45, "width_ratios": [1, 1, 0.9, 1]})
    ax = axs[0]
    ax.plot(ep, log[:, 1], color=ORANGE, marker="o", ms=2.2, label="Train")
    ax.plot(ep, log[:, 3], color=GREEN, marker="s", ms=2.2, label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss"); ax.legend()
    ax.set_xticks([1, 5, 10, 15])
    sublabel(ax, "(a) Loss")
    ax = axs[1]
    ax.plot(ep, log[:, 2], color=ORANGE, marker="o", ms=2.2, label="Train")
    ax.plot(ep, log[:, 4], color=GREEN, marker="s", ms=2.2, label="Validation")
    best = int(np.argmax(log[:, 4]))
    ax.annotate(f"best {log[best, 4]:.1f}%", (ep[best], log[best, 4]), xytext=(ep[best] - 7, log[best, 4] + 2),
                fontsize=6, arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.legend(loc="lower right")
    ax.set_xticks([1, 5, 10, 15]); ax.set_ylim(45, 95)
    sublabel(ax, "(b) Accuracy")
    ax = axs[2]
    cm = np.array(V["cnn"]["cm"])
    pct = cm / cm.sum(1, keepdims=True)
    heatmap(ax, pct, ["Healthy", "Blight"], ["Healthy", "Blight"], GREENS, ".2f", 0, 1,
            annot=[[f"{cm[i, j]}\n({pct[i, j]*100:.0f}%)" for j in range(2)] for i in range(2)])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    sublabel(ax, "(c) Confusion matrix")
    ax = axs[3]
    ax.plot(V["cnn"]["roc_fpr"], V["cnn"]["roc_tpr"], color=GREEN, lw=1.4, label=f"AUC = {V['cnn']['auc']:.3f}")
    ax.fill_between(V["cnn"]["roc_fpr"], V["cnn"]["roc_tpr"], color=GREEN, alpha=0.12, step=None)
    ax.plot([0, 1], [0, 1], color=GREY, ls=":", lw=0.8, label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    sublabel(ax, "(d) ROC curve")
    out(fig, "fig_cnn")


# --------------------------------------------------------------------------- Fig: ablation (single column)
def fig_ablation(A, B):
    names = list(A.keys())
    short = {"Full model (ours)": "Full (ours)", "w/o augmentation": "w/o aug.", "w/o dropout": "w/o dropout",
             "w/o batch normalisation": "w/o BN", "w/o augmentation & dropout": "w/o aug.\n& dropout"}
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    x = np.arange(len(names))
    tr = [A[n]["train_mean"] * 100 for n in names]
    va = [A[n]["val_mean"] * 100 for n in names]
    ax.bar(x - 0.19, tr, 0.36, yerr=[A[n]["train_std"] * 100 for n in names], color=LIGHT, edgecolor=GREEN, lw=0.6,
           error_kw=dict(lw=0.6, capsize=1.5), label="Train")
    ax.bar(x + 0.19, va, 0.36, yerr=[A[n]["val_std"] * 100 for n in names], color=GREEN, error_kw=dict(lw=0.6, capsize=1.5), label="Validation")
    for xi, v in zip(x, va):
        ax.text(xi + 0.19, 56, f"{v:.1f}", ha="center", va="bottom", fontsize=5.6, color="white", rotation=90)
    hb = B["leaf"]["Colour hist. + LogReg"]["acc"] * 100
    ax.axhline(hb, color=ORANGE, ls="--", lw=0.9)
    ax.text(-0.45, hb + 1.0, f"colour-histogram + LogReg baseline ({hb:.1f}%)", color=ORANGE, fontsize=5.6, ha="left")
    ax.set_xticks(x, [short[n] for n in names])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(50, 100)
    ax.legend(loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.16))
    ax.grid(axis="x", visible=False)
    out(fig, "fig_ablation")


# --------------------------------------------------------------------------- Fig: crop recommender
def fig_crop(V, B):
    c = V["crop"]
    labels = [l.capitalize() for l in c["labels"]]
    fig = plt.figure(figsize=(DCOL, 1.85))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1.25, 1.1, 1.1], wspace=0.5)
    ax = fig.add_subplot(gs[0])
    cm = np.array(c["cm"])
    pct = cm / cm.sum(1, keepdims=True)
    heatmap(ax, pct, [l[:4] + "." if len(l) > 5 else l for l in labels], labels, GREENS, ".2f", 0, 1,
            annot=[[f"{v*100:.0f}" if v > 0 else "" for v in r] for r in pct])
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Predicted")
    sublabel(ax, "(a) Confusion matrix (%)", y=-0.42)

    ax = fig.add_subplot(gs[1])
    x = np.arange(len(labels))
    for i, (key, col, nm) in enumerate([("precision", BLUE, "Precision"), ("recall", ORANGE, "Recall"), ("f1", GREEN, "F1")]):
        ax.bar(x + (i - 1) * 0.26, c[key], 0.26, color=col, label=nm)
    ax.set_xticks(x, [l[:4] + "." if len(l) > 5 else l for l in labels], rotation=45)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), handlelength=1)
    ax.grid(axis="x", visible=False)
    sublabel(ax, "(b) Per-class metrics", y=-0.42)

    ax = fig.add_subplot(gs[2])
    imp = c["importance"]
    order = sorted(imp, key=imp.get)
    pretty = {"N": "Nitrogen", "P": "Phosphorus", "K": "Potassium", "temperature": "Temperature", "humidity": "Humidity", "ph": "pH", "rainfall": "Rainfall"}
    ax.barh(range(len(order)), [imp[k] * 100 for k in order], color=[GREEN if imp[k] > 0.2 else TEAL if imp[k] > 0.08 else "#9DBFAF" for k in order], height=0.65)
    for i, k in enumerate(order):
        ax.text(imp[k] * 100 + 0.6, i, f"{imp[k]*100:.1f}", va="center", fontsize=5.8)
    ax.set_yticks(range(len(order)), [pretty[k] for k in order])
    ax.set_xlabel("Gini importance (%)")
    ax.set_xlim(0, 33)
    ax.grid(axis="y", visible=False)
    sublabel(ax, "(c) Feature importance", y=-0.42)

    ax = fig.add_subplot(gs[3])
    sw = [s for s in B["crop_depth_sweep"] if s["depth"] > 0]
    d = [s["depth"] for s in sw]
    ax.plot(d, [s["train"] * 100 for s in sw], color=ORANGE, marker="o", ms=2, label="Train")
    ax.plot(d, [s["val"] * 100 for s in sw], color=GREEN, marker="s", ms=2, label="Validation")
    ax.axvline(7, color=GREY, ls=":", lw=0.8)
    ax.text(6.6, 66, "chosen\ndepth = 7", fontsize=5.8, color=GREY, ha="right")
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower right")
    sublabel(ax, "(d) Depth sensitivity", y=-0.42)
    out(fig, "fig_crop")


# --------------------------------------------------------------------------- Fig: regression + forecasting
def fig_regression(V, B):
    L, F = V["lettuce"], V["forecast"]
    fig = plt.figure(figsize=(DCOL, 1.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 2.3], wspace=0.38)
    ax = fig.add_subplot(gs[0])
    yt, yp = np.array(L["y_true"]), np.array(L["y_pred"])
    lim = [12, 27]
    ax.fill_between(lim, [lim[0] - L["rmse"], lim[1] - L["rmse"]], [lim[0] + L["rmse"], lim[1] + L["rmse"]], color=GREEN, alpha=0.12, lw=0, label="±1 RMSE")
    ax.plot(lim, lim, color=GREY, ls="--", lw=0.8, label="Ideal")
    ax.scatter(yt, yp, s=6, color=GREEN, alpha=0.75, edgecolor="none")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual (days)"); ax.set_ylabel("Predicted (days)")
    ax.text(12.6, 25.4, f"$R^2$ = {L['r2']:.3f}\nMAE = {L['mae']:.2f} d", fontsize=6)
    ax.legend(loc="lower right", handlelength=1.2)
    ax.set_aspect("equal")
    sublabel(ax, "(a) Growth predictor", y=-0.36)

    ax = fig.add_subplot(gs[1])
    res = yp - yt
    ax.hist(res, bins=16, color=TEAL, edgecolor="white", lw=0.4, density=True, alpha=0.85)
    xs = np.linspace(-3.5, 3.5, 200)
    ax.plot(xs, np.exp(-0.5 * (xs / res.std()) ** 2) / (res.std() * np.sqrt(2 * np.pi)), color=ORANGE, lw=1)
    ax.text(0.03, 0.97, "Gaussian fit\n" + rf"$\sigma$ = {res.std():.2f} d", transform=ax.transAxes, va="top", fontsize=6, color=ORANGE)
    ax.axvline(0, color=GREY, lw=0.6)
    ax.set_xlabel("Residual (days)"); ax.set_ylabel("Density")
    sublabel(ax, "(b) Residual distribution", y=-0.36)

    ax = fig.add_subplot(gs[2])
    idx = pd.to_datetime(F["index"])[-288:]
    yt, yp = np.array(F["y_true"])[-288:], np.array(F["y_pred"])[-288:]
    hrs = (idx - idx[0]).total_seconds() / 3600
    ax.fill_between(hrs, yt, yp, color=ORANGE, alpha=0.18, lw=0, label="Absolute error")
    ax.plot(hrs, yt, color=GREEN, lw=0.8, label="Observed $T_{t+6}$")
    ax.plot(hrs, yp, color=ORANGE, lw=1.1, ls="--", label="Forecast (made at $t$)")
    ax.set_xlim(0, hrs.max())
    ax.set_xticks(range(0, 49, 6))
    ax.set_xlabel("Time into the last two validation days (h)")
    ax.set_ylabel("Air temperature (°C)")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.2), handlelength=1.5)
    ax.text(0.99, 0.04, f"RMSE = {F['rmse']:.2f} °C   $R^2$ = {F['r2']:.3f}", transform=ax.transAxes, ha="right", fontsize=6)
    sublabel(ax, "(c) One-hour-ahead greenhouse temperature forecast", y=-0.36)
    out(fig, "fig_regression")


# --------------------------------------------------------------------------- Fig: retrieval (single column)
def fig_retrieval(R):
    rr = R["retrieval"]
    keys = ["BM25 (question)", "TF-IDF (question)", "TF-IDF (question + answer)", "BM25 (question + answer)",
            "MiniLM dense (question) - ours", "MiniLM dense (question + answer)"]
    short = ["BM25 (Q)", "TF-IDF (Q)", "TF-IDF (Q+A)", "BM25 (Q+A)", "MiniLM (Q) — deployed", "MiniLM (Q+A)"]
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.85), gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.12})
    ax = axs[0]
    y = np.arange(len(keys))
    ax.barh(y + 0.27, [rr[k]["r1"] for k in keys], 0.27, color=GREEN, label="R@1")
    ax.barh(y, [rr[k]["r3"] for k in keys], 0.27, color=TEAL, label="R@3")
    ax.barh(y - 0.27, [rr[k]["mrr"] for k in keys], 0.27, color=LIGHT, edgecolor=GREEN, lw=0.5, label="MRR")
    for yi, k in zip(y, keys):
        ax.text(rr[k]["r1"] + 0.015, yi + 0.27, f"{rr[k]['r1']:.2f}", va="center", fontsize=5.2)
    ax.set_yticks(y, short, fontsize=6)
    ax.get_yticklabels()[4].set_fontweight("bold")
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.45, 1.17), handlelength=1, columnspacing=0.8)
    ax.grid(axis="y", visible=False)
    sublabel(ax, "(a) Retrieval quality")
    ax = axs[1]
    hit, miss = R["minilm_top_scores"]["hit"], R["minilm_top_scores"]["miss"]
    bins = np.linspace(0.2, 1.0, 17)
    ax.hist(hit, bins=bins, color=GREEN, alpha=0.85, label="correct")
    ax.hist(miss, bins=bins, color=ORANGE, alpha=0.85, label="wrong")
    ax.axvline(0.35, color=GREY, ls=":", lw=0.8)
    ax.set_xlabel("Top-1 cosine score")
    ax.yaxis.tick_right()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=2, handlelength=1, columnspacing=0.6)
    sublabel(ax, "(b) Score of top-1 match")
    out(fig, "fig_retrieval")


# --------------------------------------------------------------------------- Fig: agronomic engine (port of frontend/src/lib/agronomy.ts)
def es(t):
    return 0.61078 * np.exp(17.27 * t / (t + 237.3))


def vpd(t, rh, off=1.5):
    return np.maximum(0, es(t - off) - es(t) * rh / 100)


def suit(v, opt, tol):
    return np.exp(-0.5 * ((v - opt) / tol) ** 2)


def band(v, lo, hi, tol):
    d = np.where(v < lo, lo - v, np.where(v > hi, v - hi, 0))
    return np.exp(-0.5 * (d / tol) ** 2)


LETTUCE = dict(tempOpt=20, tempTol=5, vpdLo=0.6, vpdHi=1.0, dliOpt=15, dliTol=6, ecOpt=1.2, ecTol=0.6, co2Opt=900, co2Tol=500, baseDays=35)


def growth(p, temp, rh, ppfd, hours, ec, co2):
    s = [suit(temp, p["tempOpt"], p["tempTol"]), band(vpd(temp, rh), p["vpdLo"], p["vpdHi"], 0.35),
         suit(ppfd * hours * 3600 / 1e6, p["dliOpt"], p["dliTol"]), suit(ec, p["ecOpt"], p["ecTol"]),
         np.where(co2 >= p["co2Opt"], 1.0, band(co2, p["co2Opt"], p["co2Opt"] + p["co2Tol"], p["co2Tol"]))]
    s = np.broadcast_arrays(*s)
    gi = 0.55 * np.minimum.reduce(s) + 0.45 * np.prod(s, axis=0) ** (1 / 5)
    return gi, s


def fig_agronomy():
    fig = plt.figure(figsize=(DCOL, 1.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    T, RH = np.meshgrid(np.linspace(15, 35, 201), np.linspace(30, 95, 201))
    V = vpd(T, RH)
    zones = LinearSegmentedColormap.from_list("z", ["#7FB7E6", "#9ED9C3", "#A9D98B", "#F2C46D", "#E58B7B"], N=5)
    zi = np.digitize(V, [0.4, 0.8, 1.2, 1.6])
    ax.pcolormesh(T, RH, zi, cmap=zones, shading="auto", vmin=-0.5, vmax=4.5, rasterized=True)
    cs = ax.contour(T, RH, V, levels=[0.4, 0.8, 1.2, 1.6], colors=INK_C, linewidths=0.5)
    ax.clabel(cs, fmt="%.1f", fontsize=5.5)
    for (t, rh, lab) in [(17, 88, "Over-\ntranspiration"), (21.5, 72, "Propagation"), (25, 60, "Vegetative"), (29, 50, "Flower/\nfruit"), (32, 36, "Water\nstress")]:
        ax.text(t, rh, lab, fontsize=5.4, ha="center", va="center", color=INK_C)
    ax.set_xlabel("Air temperature (°C)"); ax.set_ylabel("Relative humidity (%)")
    ax.grid(False)
    sublabel(ax, "(a) VPD zones (kPa)")

    ax = fig.add_subplot(gs[1])
    Tt, D = np.meshgrid(np.linspace(10, 32, 160), np.linspace(4, 30, 160))
    gi, _ = growth(LETTUCE, Tt, 65, D * 1e6 / (16 * 3600), 16, 1.2, 900)
    im = ax.pcolormesh(Tt, D, gi, cmap=GREENS, shading="auto", vmin=0, vmax=1, rasterized=True)
    cs = ax.contour(Tt, D, gi, levels=[0.5, 0.7, 0.88], colors="white", linewidths=0.6)
    ax.clabel(cs, fmt="%.2f", fontsize=5.5)
    ax.plot(20, 15, marker="*", color=ORANGE, ms=6, mec="white", mew=0.4)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=5.5, length=1.5)
    cb.outline.set_linewidth(0.4)
    ax.set_xlabel("Air temperature (°C)"); ax.set_ylabel("DLI (mol m$^{-2}$ d$^{-1}$)")
    ax.grid(False)
    sublabel(ax, "(b) Lettuce growth index G")

    ax = fig.add_subplot(gs[2])
    scen = dict(temp=27, rh=55, ppfd=250, hours=16, ec=1.9, co2=600)
    gi, s = growth(LETTUCE, **scen)
    names = ["Temp.", "VPD", "DLI", "EC", "CO$_2$"]
    vals = [float(v) for v in s]
    lim = int(np.argmin(vals))
    ax.barh(range(5), vals, color=[ORANGE if i == lim else GREEN for i in range(5)], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=5.8)
    ax.axvline(float(gi), color=GREY, ls="--", lw=0.8)
    ax.text(float(gi) + 0.02, 4.55, f"G = {float(gi):.2f}", fontsize=5.8, color=GREY)
    ax.set_yticks(range(5), names)
    ax.set_xlim(0, 1.18); ax.set_ylim(-0.6, 5.0)
    ax.invert_yaxis()
    ax.set_xlabel("Factor suitability")
    ax.grid(axis="y", visible=False)
    sublabel(ax, "(c) Limiting-factor example")
    out(fig, "fig_agronomy")
    days = round(LETTUCE["baseDays"] * (1 + (1 - float(gi)) * 0.8))
    json.dump({"scenario": scen, "scores": dict(zip(["temp", "vpd", "dli", "ec", "co2"], vals)), "G": float(gi),
               "vpd": float(vpd(27, 55)), "dli": 250 * 16 * 3600 / 1e6, "days": days},
              open(os.path.join(C.RESULTS_DIR, "agronomy_example.json"), "w"), indent=2)


INK_C = "#1F2A24"


def fig_ui():
    shots = [("doctor", "(a) Plant Doctor: ViT diagnosis with treatment plan"), ("advisor", "(b) AI Advisor: retrieval-grounded answer"),
             ("simulator", "(c) Grow Simulator: growth index and limiting factor"), ("vpd", "(d) VPD Matrix: climate zone console")]
    fig, axs = plt.subplots(2, 2, figsize=(DCOL, 4.0), gridspec_kw={"wspace": 0.04, "hspace": 0.16})
    for ax, (name, cap) in zip(axs.flat, shots):
        im = Image.open(os.path.join(C.FIG_DIR, "screens", f"{name}.png")).convert("RGB")
        w, h = im.size
        im = im.crop((int(w * 0.19), 0, w, int(h * 0.94)))
        im.thumbnail((1700, 1700))
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.4); s.set_color(GREY)
        ax.text(0.5, -0.03, cap, transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
    fig.savefig(os.path.join(C.FIG_DIR, "fig_ui.png"), bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close(fig)
    print("  figure: fig_ui")


if __name__ == "__main__":
    fig_ui()
    fig_agronomy()
    V = load("verify")
    B = load("baselines")
    fig_datasets(V)
    fig_cnn(V, None, B)
    fig_crop(V, B)
    fig_regression(V, B)
    if os.path.exists(os.path.join(C.RESULTS_DIR, "ablation.json")):
        fig_ablation(load("ablation"), B)
    if os.path.exists(os.path.join(C.RESULTS_DIR, "retrieval.json")):
        fig_retrieval(load("retrieval"))
