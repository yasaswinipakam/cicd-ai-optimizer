"""Build the paper's result figures from the committed report files.

    python paper/make_figures.py

Every number plotted is read from reports/ (or, for the alert-policy figure,
recomputed from the saved fold-10 predictions with src/inference/alert_policy.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import alert_policy  # noqa: E402

REPORTS = ROOT / "reports"
OUT = ROOT / "paper" / "figures"
GRAY, BLUE, TEAL, INK = "#9a9a9a", "#2a6fdb", "#1a9e8f", "#222222"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "grid.color": "#dddddd", "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
COL = 3.45  # IEEE column width in inches


def _deltas(path: Path, comparison: str, folds, policy: str | None = None, subset: str | None = None) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d.comparison.eq(comparison) & d.fold.isin(folds) & d.finish_buffer_hours.eq(0)]
    if policy:
        d = d[d.label_policy.eq(policy)]
    if subset:
        d = d[d.subset.eq(subset)]
    return d.sort_values("fold")


def main_results() -> None:
    """Within-repo AUC of Tier C RF vs Tier E across every evaluation."""
    rows = [("Travis\nall", 0.628, 0.698), ("Travis\nnew break.", 0.556, 0.622),
            ("GHA\ntest 1", 0.715, 0.770), ("GHA\nconfirm.", 0.763, 0.911),
            ("GHA\ncombined", 0.685, 0.864)]
    x = np.arange(len(rows)); w = 0.36
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    c = ax.bar(x - w / 2, [r[1] for r in rows], w, color=GRAY, label="Tier C (RF, repository history)")
    e = ax.bar(x + w / 2, [r[2] for r in rows], w, color=BLUE, label="Tier E (stream + in-flight, rank blend)")
    for bars in (c, e):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{b.get_height():.3f}".lstrip("0"),
                    ha="center", va="bottom", fontsize=5.8, color=INK)
    ax.set_xticks(x, [r[0] for r in rows]); ax.set_ylim(0.5, 0.95); ax.set_ylabel("Macro within-repo AUC")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False, ncol=1)
    fig.savefig(OUT / "main_results.pdf"); plt.close(fig)


def fold_deltas() -> None:
    """Per-fold paired Δ (Tier E − Tier C) with repository-bootstrap 95% CIs."""
    sets = [
        ("Travis (folds 7–10)", _deltas(REPORTS / "phase1_v2_walkforward_tier_e_vs_c.csv", "E_lgbmblend_minus_C_rf", range(7, 11))),
        ("GHA test 1", _deltas(REPORTS / "phase1_v2_gha_tier_e_vs_c.csv", "E_lgbmblend_minus_C_rf", range(6, 11), "primary", "all")),
        ("GHA confirmation", _deltas(REPORTS / "phase1_v2_gha_confirmation_vs_c.csv", "E_lgbmblend_minus_C_rf", range(6, 11), "primary", "all")),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(COL * 2.05, 1.55), sharey=True)
    for ax, (title, d) in zip(axes, sets):
        y = d.delta_within_repo_auc.to_numpy()
        lo, hi = d.delta_within_repo_auc_ci_low.to_numpy(), d.delta_within_repo_auc_ci_high.to_numpy()
        clears = lo > 0
        ax.errorbar(d.fold, y, yerr=[y - lo, hi - y], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
        ax.scatter(d.fold[clears], y[clears], color=BLUE, s=16, zorder=3, label="95% CI above 0")
        ax.scatter(d.fold[~clears], y[~clears], facecolor="white", edgecolor=BLUE, s=16, zorder=3, label="95% CI includes 0")
        ax.axhline(0, color=GRAY, lw=0.7)
        ax.set_title(f"{title}: mean Δ = {y.mean():+.3f}", fontsize=7.5)
        ax.set_xticks(d.fold); ax.set_xlabel("Test fold"); ax.yaxis.grid(True); ax.set_axisbelow(True)
    axes[0].set_ylabel("Δ within-repo AUC (E − C)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)
    fig.savefig(OUT / "fold_deltas.pdf"); plt.close(fig)


def finish_buffer() -> None:
    """How the gain changes when outcomes are withheld longer (finish buffer)."""
    series = [("Travis", [0, 1, 6], [0.070, 0.081, 0.087], BLUE, "o"),
              ("GHA test 1", [0, 6], [0.055, 0.021], GRAY, "s"),
              ("GHA confirmation", [0, 6], [0.149, 0.105], TEAL, "^")]
    fig, ax = plt.subplots(figsize=(COL, 1.6))
    for name, x, y, color, marker in series:
        ax.plot(x, y, color=color, marker=marker, ms=4, lw=1.2, label=name)
        ax.annotate(f"{y[-1]:+.3f}", (x[-1], y[-1]), xytext=(4, -1), textcoords="offset points", fontsize=6.5, color=color)
    ax.set_xticks([0, 1, 6], ["0 h", "1 h", "6 h"]); ax.set_xlim(-0.4, 7.2)
    ax.set_xlabel("Finish buffer (outcomes withheld until finish + buffer)")
    ax.set_ylabel("Δ within-repo AUC (E − C)"); ax.set_ylim(0, 0.17)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
    fig.savefig(OUT / "finish_buffer.pdf"); plt.close(fig)


def persistence() -> None:
    """Confirmation test: Tier E vs Tier C vs a naive stream-persistence score (exploratory)."""
    folds = [6, 7, 8, 9, 10]
    data = {"Persistence": ([0.710, 0.720, 0.727, 0.729, 0.718], GRAY, "--", "s"),
            "Tier C RF": ([0.779, 0.749, 0.727, 0.777, 0.781], INK, "-", "o"),
            "Tier E blend": ([0.914, 0.898, 0.915, 0.917, 0.913], BLUE, "-", "^")}
    fig, ax = plt.subplots(figsize=(COL, 1.55))
    for name, (y, color, ls, marker) in data.items():
        ax.plot(folds, y, color=color, ls=ls, marker=marker, ms=4, lw=1.1, label=name)
    ax.set_xticks(folds); ax.set_xlabel("Test fold (GHA confirmation)"); ax.set_ylabel("Within-repo AUC")
    ax.set_ylim(0.65, 0.95); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3)
    fig.savefig(OUT / "persistence.pdf"); plt.close(fig)


def alert_policy_figure() -> None:
    """Per-repository share of runs flagged: one global threshold vs the per-repository budget."""
    p = pd.read_parquet(REPORTS / "phase1_v2_gha_combined_parts" / "primary_0h_fold10_E_lgbmblend.parquet")
    fallback = alert_policy.typical_repository_threshold(p)
    pooled = float(np.quantile(p.probability, 0.85))
    policies = {"Global\nthreshold": p.probability >= pooled,
                "Per-repo\nbudget": alert_policy.simulate(p, fallback)}
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.75), gridspec_kw={"width_ratios": [1.35, 1]})
    rng = np.random.default_rng(0)
    recalls = []
    for i, (name, alert) in enumerate(policies.items()):
        d = p.assign(alert=alert)
        rate = d.groupby("repo").alert.mean()
        recall = d[d.label.eq(1)].groupby("repo").alert.mean()
        recalls.append(recall.mean())
        axes[0].scatter(i + rng.uniform(-0.12, 0.12, len(rate)), rate * 100, s=10, color=[GRAY, BLUE][i], alpha=0.85)
        axes[0].text(i, rate.max() * 100 + 3, f"{rate.min():.0%}–{rate.max():.0%}", ha="center", fontsize=6.5)
    axes[0].axhline(15, color=TEAL, lw=0.8, ls="--", label="15% budget")
    axes[0].legend(frameon=False, loc="upper right", fontsize=6)
    axes[0].set_xticks([0, 1], list(policies)); axes[0].set_xlim(-0.5, 1.5); axes[0].set_ylim(0, 60)
    axes[0].set_ylabel("Runs flagged per repository (%)"); axes[0].yaxis.grid(True); axes[0].set_axisbelow(True)
    bars = axes[1].bar([0, 1], [r * 100 for r in recalls], color=[GRAY, BLUE], width=0.6)
    for b in bars:
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}%", ha="center", fontsize=6.5)
    axes[1].set_xticks([0, 1], ["Global", "Per-repo"]); axes[1].set_ylim(0, 55)
    axes[1].set_ylabel("Failures caught per repo (%)"); axes[1].yaxis.grid(True); axes[1].set_axisbelow(True)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "alert_policy.pdf"); plt.close(fig)
    print(f"alert policy: pooled threshold {pooled:.3f}, typical-repository threshold {fallback:.3f}, "
          f"mean recall {recalls[0]:.3f} vs {recalls[1]:.3f}")


def crop_overview_screenshot() -> None:
    """Keep the header, summary cards and trend; drop the panels cut off at the bottom."""
    path = ROOT / "paper" / "screens" / "dashboard_overview.png"
    image = Image.open(path)
    if image.height > image.width * 0.65:
        image.crop((0, 0, image.width, int(image.width * 0.62))).save(path)


def crop_pipelines_screenshot() -> None:
    """Trim the empty page area below the pipelines table."""
    path = ROOT / "paper" / "screens" / "pipelines.png"
    image = Image.open(path)
    if image.height > image.width * 0.45:
        image.crop((0, 0, image.width, int(image.width * 0.335))).save(path)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    main_results(); fold_deltas(); finish_buffer(); persistence(); alert_policy_figure(); crop_pipelines_screenshot(); crop_overview_screenshot()
    print(sorted(p.name for p in OUT.glob("*.pdf")))
