"""Schematic diagrams for the VERDANT paper (architecture, CNN, RAG advisor).

    python paper/diagrams.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures as F  # noqa: E402  (shares rcParams + palette)

INK = "#1F2A24"


def box(ax, x, y, w, h, text, fc="white", ec=F.GREEN, lw=0.8, fs=6.6, bold=False, r=0.012, color=INK, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, p, q, color=INK, lw=0.8, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=lw, color=color,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def note(ax, x, y, text, **kw):
    ax.text(x, y, text, fontsize=kw.pop("fs", 5.4), color=F.GREY, style="italic", **kw)


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------------- system architecture
def architecture():
    fig, ax = canvas(F.DCOL, 2.75)
    lanes = [(0.005, 0.215, "#EEF5F1", F.GREEN, "PRESENTATION TIER", "React 19 · TypeScript · Vite · Tailwind"),
             (0.285, 0.255, "#EEF2F8", F.BLUE, "APPLICATION TIER", "FastAPI · Uvicorn · Python"),
             (0.61, 0.385, "#FBF1E7", F.ORANGE, "MODEL TIER", "PyTorch · Transformers · scikit-learn")]
    for x, w, fc, ec, title, sub in lanes:
        ax.add_patch(FancyBboxPatch((x, 0.02), w, 0.96, boxstyle="round,pad=0,rounding_size=0.015", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, 0.935, title, ha="center", va="center", fontsize=7.2, fontweight="bold", color=ec)
        ax.text(x + w / 2, 0.885, sub, ha="center", va="center", fontsize=5.9, color=F.GREY, style="italic")

    # presentation tier
    ui = ["Plant Doctor\n(leaf upload → diagnosis)", "Species Scanner\n(photo → species profile)", "AI Advisor\n(chat, related questions)",
          "Grow Planner\n(crop → climate targets)", "Simulator · VPD matrix ·\nNutrient lab · Dashboard"]
    ys = [0.72, 0.585, 0.45, 0.315, 0.18]
    for t, y in zip(ui, ys):
        box(ax, 0.025, y, 0.175, 0.105, t, ec=F.GREEN, fs=6.1)
    note(ax, 0.11, 0.075, "client-side agronomic engine (Sec. III-D)", ha="center", fs=5.6)

    # application tier
    eps = [("POST /api/diagnose", 0.745), ("POST /api/identify", 0.61), ("POST /api/advisor", 0.475), ("POST /api/growplan", 0.34), ("GET  /api/health", 0.205)]
    for t, y in eps:
        box(ax, 0.305, y, 0.215, 0.07, t, ec=F.BLUE, fs=6.3)
    box(ax, 0.305, 0.045, 0.215, 0.12, "Lazy model cache · per-model locks\nbackground warm-up thread\nasync LLM client · input validation",
        fc="#DCE6F3", ec=F.BLUE, fs=5.8, lw=0.6)
    for (_, ye), yu in zip(eps, ys):
        arrow(ax, (0.2, yu + 0.0525), (0.305, ye + 0.035), color=F.GREY, lw=0.6)
    note(ax, 0.2525, 0.83, "HTTP / JSON\nvia /api proxy", ha="center", fs=5.6)

    # model tier
    def group(x, y, w, h, title):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.01", fc="white", ec=F.ORANGE, lw=0.6, ls="--"))
        ax.text(x + 0.008, y + h - 0.012, title, ha="left", va="top", fontsize=5.9, color=F.ORANGE, fontweight="bold")

    group(0.625, 0.535, 0.175, 0.32, "Pretrained (HF Hub)")
    box(ax, 0.635, 0.715, 0.155, 0.085, "ViT-Tiny disease\nclassifier (13)", ec=F.ORANGE, fs=6.1)
    box(ax, 0.635, 0.62, 0.155, 0.085, "ViT-Base species\nclassifier (52)", ec=F.ORANGE, fs=6.1)
    box(ax, 0.635, 0.545, 0.155, 0.065, "MiniLM-L6 encoder", ec=F.ORANGE, fs=6.1)

    group(0.81, 0.535, 0.175, 0.32, "Trained from scratch")
    for t, y in [("Foliar disease CNN", 0.76), ("RF crop recommender", 0.695), ("Ridge growth model", 0.63), ("RF climate forecaster", 0.565)]:
        box(ax, 0.82, y, 0.155, 0.052, t, ec=F.ORANGE, fs=6.0)
    note(ax, 0.8975, 0.543, "offline evaluation (Sec. VI)", ha="center", fs=5.0)

    group(0.625, 0.05, 0.36, 0.44, "Knowledge & generation")
    box(ax, 0.64, 0.29, 0.155, 0.125, "Gemma 3 LLM\n(Ollama API)", fc="#FFF8F0", ec=F.ORANGE, fs=6.2, ls="--")
    box(ax, 0.815, 0.29, 0.155, 0.125, "Agronomy KB\n34 Q–A pairs\n12 topics", ec=F.ORANGE, fs=6.1)
    box(ax, 0.64, 0.085, 0.155, 0.16, "Template fallback\n(retrieved answer /\ndefault grow plan)", ec=F.GREY, fs=5.9, ls=":")
    box(ax, 0.815, 0.085, 0.155, 0.16, "KB embeddings\n34 × 384\n(cached at start-up)", ec=F.ORANGE, fs=6.0)
    arrow(ax, (0.7175, 0.29), (0.7175, 0.245), color=F.GREY, lw=0.6, ls=":")
    note(ax, 0.722, 0.262, "if unavailable", va="center", fs=5.2)
    arrow(ax, (0.8925, 0.29), (0.8925, 0.245), color=F.GREY, lw=0.6)
    arrow(ax, (0.815, 0.3525), (0.795, 0.3525), color=F.ORANGE, lw=0.7)
    note(ax, 0.805, 0.37, "context", ha="center", fs=5.0)
    arrow(ax, (0.7125, 0.545), (0.8925, 0.415), color=F.ORANGE, lw=0.6, rad=-0.25)
    note(ax, 0.86, 0.47, "top-1 retrieval", ha="center", fs=5.0)

    # API -> models
    arrow(ax, (0.52, 0.78), (0.635, 0.7575), color=F.BLUE, lw=0.7)
    arrow(ax, (0.52, 0.645), (0.635, 0.6625), color=F.BLUE, lw=0.7)
    arrow(ax, (0.52, 0.51), (0.635, 0.5775), color=F.BLUE, lw=0.7)
    arrow(ax, (0.52, 0.5), (0.64, 0.38), color=F.BLUE, lw=0.7)
    arrow(ax, (0.52, 0.375), (0.64, 0.33), color=F.BLUE, lw=0.7)
    F.out(fig, "fig_architecture")


# --------------------------------------------------------------------------- CNN schematic
def cuboid(ax, x, y, w, h, d, fc, ec=INK, lw=0.5):
    """Pseudo-3D block: front face (w x h) with depth d drawn up-right."""
    dx, dy = d * 0.55, d * 0.42
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], closed=True, fc=fc, ec=ec, lw=lw))
    ax.add_patch(Polygon([(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)], closed=True, fc=fc, ec=ec, lw=lw, alpha=0.75))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)], closed=True, fc=fc, ec=ec, lw=lw, alpha=0.55))
    return x + w + dx


def cnn_schematic():
    fig, ax = canvas(F.DCOL, 1.55)
    base = 0.2
    blocks = [(0.012, 0.56, 0.11, "#C9D6CF", "Input", "3×128×128"),
              (0.022, 0.42, 0.085, "#A8D5BA", "Block 1", "16×64×64"),
              (0.034, 0.3, 0.06, "#7FC29B", "Block 2", "32×32×32"),
              (0.05, 0.2, 0.04, "#4FA77A", "Block 3", "64×16×16"),
              (0.072, 0.12, 0.025, "#2E7D5B", "Block 4", "128×8×8")]
    x = 0.02
    centers = []
    for w, h, d, c, top, bot in blocks:
        y = base + (0.56 - h) / 2
        right = cuboid(ax, x, y, w, h, d, c)
        centers.append((x, right))
        ax.text((x + right) / 2, 0.94, top, ha="center", va="top", fontsize=6.4, fontweight="bold", color=INK)
        ax.text((x + right) / 2, 0.08, bot, ha="center", va="center", fontsize=6.0, color=F.GREY)
        x = right + 0.075
    for (x0, r0), (x1, r1) in zip(centers[:-1], centers[1:]):
        arrow(ax, (r0 + 0.006, base + 0.28), (x1 - 0.006, base + 0.28), lw=0.7)
        ax.text((r0 + x1) / 2, base + 0.36, "Conv 3×3\nBN · ReLU\nMaxPool 2×2", ha="center", va="bottom", fontsize=4.9, color=F.GREY, linespacing=1.05)

    xs = x - 0.075 + 0.04
    arrow(ax, (centers[-1][1] + 0.006, base + 0.28), (xs, base + 0.28), lw=0.7)
    heads = [("8192", 0.62, "#DDE7E1", "Flatten\n+ Dropout 0.4"), ("128", 0.4, "#9FC9B2", "FC + ReLU"), ("2", 0.14, "#E07A1F", "FC + Softmax")]
    for i, (n, h, c, lab) in enumerate(heads):
        y = base + (0.56 - h) / 2
        ax.add_patch(FancyBboxPatch((xs, y), 0.022, h, boxstyle="round,pad=0,rounding_size=0.004", fc=c, ec=INK, lw=0.5))
        ax.text(xs + 0.011, 0.94, lab, ha="center", va="top", fontsize=6.0, fontweight="bold", color=INK, linespacing=1.05)
        ax.text(xs + 0.011, 0.08, n, ha="center", va="center", fontsize=6.0, color=F.GREY)
        if i < 2:
            arrow(ax, (xs + 0.026, base + 0.28), (xs + 0.084, base + 0.28), lw=0.7)
        xs += 0.088
    ax.text(xs + 0.005, base + 0.34, "Healthy", fontsize=6.2, va="center", color=F.GREEN)
    ax.text(xs + 0.005, base + 0.22, "Late blight", fontsize=6.2, va="center", color=F.ORANGE)
    arrow(ax, (xs - 0.062, base + 0.3), (xs - 0.001, base + 0.34), lw=0.5, color=F.GREY)
    arrow(ax, (xs - 0.062, base + 0.26), (xs - 0.001, base + 0.22), lw=0.5, color=F.GREY)
    F.out(fig, "fig_cnn_arch")


# --------------------------------------------------------------------------- RAG advisor
def rag():
    fig, ax = canvas(F.COL, 2.85)
    ax.add_patch(FancyBboxPatch((0.01, 0.80), 0.98, 0.185, boxstyle="round,pad=0,rounding_size=0.02", fc="#F3F3F3", ec="#BDBDBD", lw=0.6, ls="--"))
    note(ax, 0.03, 0.968, "Offline (server start-up)", va="top", fs=6)
    box(ax, 0.04, 0.815, 0.25, 0.11, "Knowledge base\n34 Q–A, 12 topics", ec=F.GREY, fs=6.0)
    box(ax, 0.375, 0.815, 0.25, 0.11, "MiniLM-L6-v2\nencoder", ec=F.ORANGE, fs=6.0)
    box(ax, 0.71, 0.815, 0.25, 0.11, "Embeddings\n" + r"$K \in \mathbb{R}^{34 \times 384}$", ec=F.ORANGE, fs=6.0)
    arrow(ax, (0.29, 0.87), (0.375, 0.87), lw=0.7)
    arrow(ax, (0.625, 0.87), (0.71, 0.87), lw=0.7)

    note(ax, 0.03, 0.765, "Online (per request)", fs=6)
    box(ax, 0.04, 0.635, 0.25, 0.11, r"User question $q$", ec=F.GREEN, fs=6.2, fc="#EEF5F1")
    box(ax, 0.375, 0.635, 0.25, 0.11, "MiniLM-L6-v2\nencoder", ec=F.ORANGE, fs=6.0)
    box(ax, 0.71, 0.635, 0.25, 0.11, "Cosine similarity\n" + r"$s_i = \hat{q}^{\top}\hat{k}_i$, top-1", ec=F.BLUE, fs=6.0)
    arrow(ax, (0.29, 0.69), (0.375, 0.69), lw=0.7)
    arrow(ax, (0.625, 0.69), (0.71, 0.69), lw=0.7)
    arrow(ax, (0.835, 0.815), (0.835, 0.745), lw=0.7, color=F.ORANGE)

    box(ax, 0.62, 0.45, 0.34, 0.11, r"$s^{*} < 0.35$?  prepend" + "\nlow-confidence notice", ec=F.GREY, fs=5.8, ls=":")
    arrow(ax, (0.835, 0.635), (0.835, 0.56), lw=0.7)
    box(ax, 0.04, 0.45, 0.45, 0.11, "Prompt = system role + retrieved\nanswer (context) + question", ec=F.BLUE, fs=5.9)
    arrow(ax, (0.62, 0.505), (0.49, 0.505), lw=0.7)

    box(ax, 0.04, 0.24, 0.45, 0.13, "Gemma 3 (4B) via Ollama\n3–5 sentence grounded answer", ec=F.ORANGE, fs=6.0, fc="#FFF8F0")
    arrow(ax, (0.265, 0.45), (0.265, 0.37), lw=0.7)
    box(ax, 0.58, 0.24, 0.38, 0.13, "Fallback: return the\nretrieved KB answer", ec=F.GREY, fs=5.9, ls=":")
    arrow(ax, (0.49, 0.305), (0.58, 0.305), lw=0.7, ls=":", color=F.GREY)
    note(ax, 0.535, 0.385, "no key /\ntimeout /\nparse error", ha="center", va="bottom", fs=5.0, linespacing=1.0)

    box(ax, 0.2, 0.035, 0.6, 0.12, "Response: answer · topic · score ·\n3 related questions · source", ec=F.GREEN, fs=6.0, fc="#EEF5F1")
    arrow(ax, (0.265, 0.24), (0.38, 0.155), lw=0.7)
    arrow(ax, (0.77, 0.24), (0.62, 0.155), lw=0.7, ls=":", color=F.GREY)
    F.out(fig, "fig_rag")


if __name__ == "__main__":
    architecture()
    cnn_schematic()
    rag()
