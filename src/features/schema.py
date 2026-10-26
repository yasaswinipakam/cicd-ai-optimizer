"""Canonical, leak-free model input schema.

Import ``FEATURE_COLS`` from this module everywhere model inputs are selected.
"""

# IMPORTANT: tr_log_* fields are deliberately excluded. They describe test results,
# durations, and final build state, which become available only after the build has
# completed. Including them would leak the outcome into a supposed pre-build
# prediction and make evaluation misleadingly optimistic.

# Direct numeric inputs ported from the notebook only after checking each one
# against the feature-availability registry.  These are available at push/build
# start and are deliberately separate from repository identity (gh_project_name),
# which is not a prediction feature.
DIRECT_PREBUILD_FEATURE_COLS = [
    "git_diff_src_churn",
    "git_diff_test_churn",
    "gh_diff_files_added",
    "gh_diff_files_deleted",
    "gh_diff_files_modified",
    "gh_diff_tests_added",
    "gh_diff_tests_deleted",
    "gh_diff_src_files",
    "gh_diff_doc_files",
    "gh_diff_other_files",
    "gh_num_commits_on_files_touched",
    "gh_num_commit_comments",
    "git_num_all_built_commits",
    "gh_sloc",
    "gh_test_lines_per_kloc",
    "gh_test_cases_per_kloc",
    "gh_asserts_cases_per_kloc",
    "gh_team_size",
    "gh_repo_age",
    "gh_repo_num_commits",
    "tr_build_number",
    # Previously unused pre-build columns — all available at trigger time.
    "gh_num_commits_in_push",
    "gh_num_pr_comments",
    "gh_num_issue_comments",
]

DERIVED_PREBUILD_FEATURE_COLS = [
    "prev_status_enc",
    "lang_enc",
    "is_master",
    "by_core",
    "is_pr",
    "test_to_src_ratio",
    "total_files_changed",
    "total_test_changes",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_night",
    # Rolling per-repo failure history — finish-time-safe heap protocol.
    # Cold-start default is COLD_START_RATE (empirical global failure prior).
    "repo_fail_rate_last5",
    "repo_fail_rate_last20",
    "repo_prior_fail_rate",
    # Streak and gap features — also finish-time-safe. Cold-start default is 0.0.
    "repo_streak_fails",
    "repo_builds_since_success",
    "repo_hours_since_last_build",
]

FEATURE_COLS = DIRECT_PREBUILD_FEATURE_COLS + DERIVED_PREBUILD_FEATURE_COLS

# Names of the three rolling-history features (subset of DERIVED_PREBUILD_FEATURE_COLS).
# Callers that need to pre-compute them before splitting can import this list.
HISTORY_FEATURE_COLS = [
    "repo_fail_rate_last5",
    "repo_fail_rate_last20",
    "repo_prior_fail_rate",
]

# Minimal raw/processed columns needed to construct every strict feature.  All
# training/evaluation loaders import this list instead of silently dropping newly
# added notebook-derived fields during column-pruned parquet reads.
FEATURE_SOURCE_COLUMNS = [
    "git_prev_commit_resolution_status",
    "gh_lang",
    "git_branch",
    "gh_by_core_team_member",
    "gh_is_pr",
    "gh_build_started_at",
    *DIRECT_PREBUILD_FEATURE_COLS,
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_night",
    # Raw columns for the newly added direct pre-build features (deduped at load
    # time via dict.fromkeys, so repeats from DIRECT_PREBUILD_FEATURE_COLS are fine).
    "gh_num_commits_in_push",
    "gh_num_pr_comments",
    "gh_num_issue_comments",
]
