"""Tests for failed-build root-cause clustering."""

import pandas as pd

from src.models.clustering import RootCauseClusterer


def test_distinct_failure_themes_are_separated_and_blanks_use_own_bucket() -> None:
    database_failures = [f"database connection timeout postgres query {index}" for index in range(8)]
    auth_failures = [f"authentication token login permission denied {index}" for index in range(7)]
    logs = pd.Series(database_failures + auth_failures + [None, "  "])

    clusterer = RootCauseClusterer(n_clusters=3).fit(logs)
    labels = clusterer.predict(logs)

    assert labels[0] != labels[8]
    assert labels[-1] == clusterer.non_test_cluster
    assert labels[-2] == clusterer.non_test_cluster
    top_terms = clusterer.top_terms_per_cluster(3)
    assert top_terms[clusterer.non_test_cluster] == ["non-test failure"]
