"""Train the post-hoc failed-build root-cause clustering layer."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import matplotlib

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "cicd-ai-optimizer-matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)
os.environ["XDG_CACHE_HOME"] = str(Path(tempfile.gettempdir()) / "cicd-ai-optimizer-cache")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.models.clustering import RootCauseClusterer
from src.models.train import ARTIFACTS_DIR, DATA_PATH


FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"
CLUSTERER_PATH = ARTIFACTS_DIR / "root_cause_clusterer.joblib"


def _save_cluster_chart(counts: pd.Series, terms: dict[int, list[str]]) -> None:
    """Save cluster sizes with concise top-term labels."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = [" / ".join(terms.get(cluster, ["unknown"])[:3]) for cluster in counts.index]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(labels, counts.values, color="#4c78a8")
    axis.set_title("Failed Build Root-Cause Clusters")
    axis.set_ylabel("Failed builds")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "root_cause_clusters.png", dpi=150)
    plt.close(figure)


def train_root_cause_clusterer(data_path: Path = DATA_PATH) -> RootCauseClusterer:
    """Cluster failed builds into three test themes plus a non-test bucket."""
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}. Run the data pipeline first.")
    df = pd.read_parquet(data_path, engine="pyarrow")
    required = {"tr_status", "tr_log_tests_failed"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

    failed_logs = df.loc[df["tr_status"].eq("failed"), "tr_log_tests_failed"]
    if failed_logs.empty:
        raise ValueError("No failed builds are available for root-cause clustering.")
    # Four output buckets: three test-text clusters plus one non-test-failure bucket.
    clusterer = RootCauseClusterer(n_clusters=4).fit(failed_logs)
    labels = clusterer.predict(failed_logs)
    counts = pd.Series(labels).value_counts().reindex(range(4), fill_value=0).sort_index()
    terms = clusterer.top_terms_per_cluster()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    clusterer.save(str(CLUSTERER_PATH))
    _save_cluster_chart(counts, terms)
    print("Root-cause cluster sizes and top terms:")
    for cluster_id, count in counts.items():
        print(f"  Cluster {cluster_id}: {count:,} builds — {', '.join(terms.get(cluster_id, []))}")
    return clusterer


def main() -> None:
    """Run root-cause clustering from the command line."""
    train_root_cause_clusterer()


if __name__ == "__main__":
    main()
