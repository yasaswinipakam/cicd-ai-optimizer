// Readable names for the model inputs that appear in prediction explanations.
const LABELS = {
  git_diff_src_churn: 'Source lines changed',
  gh_diff_files_added: 'Files added',
  gh_diff_files_deleted: 'Files deleted',
  gh_diff_files_modified: 'Files modified',
  gh_diff_tests_added: 'Tests added',
  gh_diff_tests_deleted: 'Tests deleted',
  gh_diff_src_files: 'Source files changed',
  gh_diff_doc_files: 'Doc files changed',
  gh_diff_other_files: 'Other files changed',
  gh_is_pr: 'Pull request',
  is_master: 'On master branch',
  gh_by_core_team_member: 'Core-team author',
  gh_team_size: 'Team size',
  gh_sloc: 'Repository size (lines)',
  gh_num_commits_on_files_touched: 'Past commits on these files',
  git_num_all_built_commits: 'Commits in this build',
  gh_test_lines_per_kloc: 'Test lines per 1k lines',
  gh_test_cases_per_kloc: 'Test cases per 1k lines',
  gh_asserts_cases_per_kloc: 'Asserts per 1k lines',
  additions: 'Lines added',
  deletions: 'Lines deleted',
  total_churn: 'Lines changed',
  files_modified: 'Files changed',
  msg_len: 'Commit message length',
  is_merge: 'Merge commit',
  num_parents: 'Commit parents',
  time_since_last_commit: 'Seconds since previous commit',
  is_pr: 'Pull request',
  is_default_branch: 'On default branch',
  hour: 'Hour of day',
  day_of_week: 'Day of week',
  month: 'Month',
  is_weekend: 'Weekend',
  is_night: 'Night-time',
  hist_prev_failed: 'Last pipeline build failed',
  hist_repo_fail_rate_5: 'Pipeline failure rate (last 5)',
  hist_repo_fail_rate_all: 'Pipeline failure rate (all)',
  hist_repo_ewma: 'Recent pipeline failure trend',
  hist_stream_prev_failed: 'Last build on branch/PR failed',
  hist_stream_fail_rate_5: 'Branch/PR failure rate (last 5)',
  hist_stream_fail_rate_all: 'Branch/PR failure rate (all)',
  hist_stream_completed: 'Finished builds on branch/PR',
  hist_stream_streak: 'Failure streak on branch/PR',
  hist_prevbuild_failed: 'Previous run failed',
  ctx_prevbuild_pending: 'Previous build still running',
  ctx_stream_pending: 'Branch/PR build still running',
  ctx_repo_pending: 'Builds running in pipeline',
  ctx_repo_started_1h: 'Builds started in last hour',
  ctx_hours_since_stream_start: 'Hours since last branch/PR build',
}

const featureLabel = feature => LABELS[feature] ?? feature

// ctx_repo_pending is a count, so only the two single-build "pending" flags read as yes/no.
const YES_NO = /(^is_|_failed$|^ctx_(prevbuild|stream)_pending$|^gh_is_pr$|^gh_by_core_team_member$)/
const RATE = /(rate|ewma)/

// "Label: value", with yes/no for flags and percentages for rates.
export function featureWithValue(feature, value) {
  if (value == null || Number.isNaN(value)) return featureLabel(feature)
  const shown = YES_NO.test(feature) ? (value ? 'yes' : 'no')
    : RATE.test(feature) ? `${Math.round(value * 100)}%`
    : Number.isInteger(value) ? value.toLocaleString() : value.toFixed(1)
  return `${featureLabel(feature)}: ${shown}`
}
