"""Post-hoc root-cause clustering from failed-test names."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class RootCauseClusterer:
    """Group failed builds by their failing-test text.

    ``n_clusters`` is the total number of output buckets. One is reserved for
    failures without test text, leaving ``n_clusters - 1`` text-derived clusters.
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2 (one text cluster and one non-test bucket).")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.non_test_cluster = n_clusters - 1
        self.vectorizer: TfidfVectorizer | None = None
        self.kmeans: KMeans | None = None
        self.test_cluster_count = 0

    @staticmethod
    def _non_test_mask(failed_logs: pd.Series) -> pd.Series:
        """Identify missing or whitespace-only entries as non-test failures."""
        return failed_logs.isna() | failed_logs.astype(str).str.strip().eq("")

    def fit(self, failed_logs: pd.Series) -> "RootCauseClusterer":
        """Fit TF-IDF and KMeans only on records containing failing-test text."""
        logs = pd.Series(failed_logs, copy=False)
        non_test = self._non_test_mask(logs)
        test_logs = logs.loc[~non_test].astype(str)
        self.vectorizer = None
        self.kmeans = None
        self.test_cluster_count = 0
        if test_logs.empty:
            return self

        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        matrix = vectorizer.fit_transform(test_logs)
        self.test_cluster_count = min(self.n_clusters - 1, len(test_logs))
        self.vectorizer = vectorizer
        self.kmeans = KMeans(
            n_clusters=self.test_cluster_count,
            random_state=self.random_state,
            n_init=10,
        ).fit(matrix)
        return self

    def predict(self, failed_logs: pd.Series) -> np.ndarray:
        """Assign text logs to KMeans clusters and blanks to the non-test bucket."""
        if self.vectorizer is None or self.kmeans is None:
            return np.full(len(failed_logs), self.non_test_cluster, dtype=int)

        logs = pd.Series(failed_logs, copy=False)
        non_test = self._non_test_mask(logs)
        labels = np.full(len(logs), self.non_test_cluster, dtype=int)
        test_positions = np.flatnonzero(~non_test.to_numpy())
        if len(test_positions):
            matrix = self.vectorizer.transform(logs.iloc[test_positions].astype(str))
            labels[test_positions] = self.kmeans.predict(matrix)
        return labels

    def top_terms_per_cluster(self, n_terms: int = 10) -> dict[int, list[str]]:
        """Return human-readable top terms from each cluster centroid."""
        terms: dict[int, list[str]] = {self.non_test_cluster: ["non-test failure"]}
        if self.vectorizer is None or self.kmeans is None:
            return terms

        vocabulary = self.vectorizer.get_feature_names_out()
        for cluster_id, centroid in enumerate(self.kmeans.cluster_centers_):
            top_indices = centroid.argsort()[-n_terms:][::-1]
            terms[cluster_id] = vocabulary[top_indices].tolist()
        return terms

    def save(self, path: str) -> None:
        """Persist the fitted clusterer."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "RootCauseClusterer":
        """Load an artifact produced by :meth:`save`."""
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Artifact at {path} is not a {cls.__name__}.")
        return loaded
