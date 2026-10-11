# Cross-Repository Generalization Evaluation

The raw field `gh_project_name` identifies **947** repositories across **3,562,314** builds. Repository sizes are highly imbalanced: minimum=1, median=1,130, maximum=132,888; the largest repository accounts for 3.73% of builds and the three largest account for 9.64%.

The three largest repositories were held out one at a time using a predeclared build-count rule: `DataDog/dd-agent` (132,888), `pyca/cryptography` (108,767), `opf/openproject` (101,576). This avoids selecting repositories based on favourable performance. `gh_project_name` is not in `FEATURE_COLS` and was removed before feature engineering, so no repository identity enters a model matrix.

| Evaluation | Test builds | Test failure rate | Precision | Recall | F1 | AUC | Top test language |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| In-distribution random reference | 40,000 | 0.2116 | 0.3286 | 0.6217 | 0.4300 | 0.6902 | Mixed |
| Held-out: DataDog/dd-agent | 132,888 | 0.5348 | 0.5247 | 0.1835 | 0.2719 | 0.4532 | python |
| Held-out: pyca/cryptography | 108,767 | 0.0334 | 0.0535 | 0.6149 | 0.0984 | 0.6366 | python |
| Held-out: opf/openproject | 101,576 | 0.3100 | 0.3478 | 0.4636 | 0.3974 | 0.5455 | ruby |

Held-out F1 ranges from 0.0984–0.3974. Compare these values with the in-distribution random reference in the table rather than interpreting any single held-out result in isolation. Differences can reflect repository-specific failure base rates, language mix, team practices, and feature-distribution shift. This experiment concerns only the selected large repositories in this TravisTorrent export and does not establish universal cross-project generalization.
