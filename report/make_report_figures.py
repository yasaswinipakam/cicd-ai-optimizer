"""Figures used only in the capstone report (the paper's figures live in paper/figures).

    python report/make_report_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
GRAY, BLUE, TEAL, RED, INK = "#9a9a9a", "#2a6fdb", "#1a9e8f", "#c0392b", "#222222"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "grid.color": "#dddddd",
    "grid.linewidth": 0.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
})


def model_comparison() -> None:
    """Development reference: six classifiers on the random 80/20 split (artifacts/model_results.csv)."""
    d = pd.read_csv(ROOT / "artifacts" / "model_results.csv")
    names = {"random_forest": "Random Forest", "decision_tree": "Decision Tree", "svm": "SVM", "xgboost": "XGBoost",
             "lightgbm": "LightGBM", "stacking": "Stacking"}
    d["name"] = d.Model.map(names)
    d = d.set_index("name").loc[["Decision Tree", "SVM", "Random Forest", "XGBoost", "LightGBM", "Stacking"]]
    metrics = [("Precision", GRAY), ("Recall", TEAL), ("F1", BLUE), ("AUC", INK)]
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    width = 0.2
    for i, (metric, color) in enumerate(metrics):
        xs = [k + (i - 1.5) * width for k in range(len(d))]
        ax.bar(xs, d[metric], width, color=color, label=metric)
    for k, auc in enumerate(d.AUC):
        ax.text(k + 1.5 * width, auc + 0.01, f"{auc:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(d)), d.index); ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(OUT / "model_comparison.pdf"); plt.close(fig)


def leakage_ladder() -> None:
    """The same task under progressively stricter evaluation (reports/final_results.md, leakage_regime_analysis.md)."""
    rows = [("Random split, 31 features incl. post-build logs (leaky)", 0.896, RED),
            ("Random split, 39 features with history (stacking)", 0.912, GRAY),
            ("Random split, 13 strict pre-build features", 0.690, GRAY),
            ("Chronological split, job level, pooled (RF)", 0.821, GRAY),
            ("Walk-forward, build level, within-repo (Tier C RF)", 0.628, BLUE),
            ("Walk-forward, build level, within-repo (Tier E)", 0.698, TEAL)]
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    ys = range(len(rows))[::-1]
    ax.barh(list(ys), [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
    for y, (_, value, _) in zip(ys, rows):
        ax.text(value + 0.005, y, f"{value:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(list(ys), [r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("AUC (0.5 = random)"); ax.xaxis.grid(True); ax.set_axisbelow(True)
    fig.savefig(OUT / "leakage_ladder.pdf"); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    model_comparison(); leakage_ladder()
    print(sorted(p.name for p in OUT.glob("*.pdf")))
