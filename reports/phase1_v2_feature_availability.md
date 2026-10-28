# Phase 1 feature-availability record

This record was fixed before running the Tier Cplus walk-forward units.

| Requested feature family | Availability in `data/processed/clean.parquet` | Phase 1 handling |
|---|---|---|
| Repository history | Build id, repository, trigger time, duration, and collapsed build outcome are present. | Implemented using only builds whose derived finish time is strictly before the focal trigger time. |
| Committer history / first-time contributor | No stable committer/account identifier is present. | Not implemented; no proxy is used. |
| File-level failure history | Only aggregate diff counts are present; no per-build changed-file paths are present. | Not implemented. |
| Change-type flags (CI/config/dependency/lock files) | No path or file-type flags are present. | Not implemented. Existing aggregate source/test/doc/other counts remain Tier A features. |

Tier Cplus adds: last-10 and last-20 completed-build failure rates, consecutive completed-build failure streak, hours since the last completed build, and hours since the last completed failed build. Tier C is retained unchanged for comparison.
