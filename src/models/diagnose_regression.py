"""Read-only diagnostic for investigating the notebook-to-pipeline F1 gap.

This script never writes or replaces production model, encoder, or preprocessing
artifacts. It prints a Markdown report that can be saved as project evidence.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.features.engineer import engineer_features
from src.models.train import ARTIFACTS_DIR, DATA_PATH, MAX_TRAINING_ROWS, RANDOM_STATE


RAW_FEATURE_COLUMNS = [
    "tr_status", "git_prev_commit_resolution_status", "gh_lang", "git_branch",
    "gh_by_core_team_member", "gh_is_pr", "git_diff_src_churn", "git_diff_test_churn",
    "gh_diff_src_files", "gh_diff_tests_added", "gh_diff_tests_deleted", "hour", "day_of_week", "month",
    "is_weekend", "is_night",
]
LEAKY_LOG_COLUMNS = [
    "tr_log_status", "tr_log_setup_time", "tr_log_bool_tests_ran", "tr_log_bool_tests_failed",
    "tr_log_num_tests_ok", "tr_log_num_tests_failed", "tr_log_num_tests_run", "tr_log_num_tests_skipped",
    "tr_log_num_test_suites_run", "tr_log_num_test_suites_ok", "tr_log_num_test_suites_failed",
    "tr_log_testduration", "tr_log_buildduration",
]


def _rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )


def _load_base_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH, columns=RAW_FEATURE_COLUMNS + LEAKY_LOG_COLUMNS)


def _sample_split(df: pd.DataFrame, feature_cols: list[str]):
    if len(df) > MAX_TRAINING_ROWS:
        df = df.sample(n=MAX_TRAINING_ROWS, random_state=RANDOM_STATE)
    target = df["tr_status"].eq("failed").astype(int)
    return train_test_split(df[feature_cols], target, test_size=0.2, stratify=target, random_state=RANDOM_STATE)


def _fallback_rate(
    df: pd.DataFrame, encoders: dict[str, LabelEncoder], feature: str, source: str
) -> tuple[int, int, float, int, float, int, float]:
    values = df[source].fillna("__missing__").astype(str) if source in df else pd.Series("__missing__", index=df.index)
    unseen = ~values.isin(encoders[feature].classes_)
    missing = values.eq("__missing__")
    return (
        # The transform-only implementation substitutes a fallback only for
        # categories not present in encoder.classes_.  A natural occurrence of
        # the encoder's first class is not a fallback event.
        int(unseen.sum()), len(values), float(unseen.mean()),
        int(unseen.sum()), float(unseen.mean()), int(missing.sum()), float(missing.mean()),
    )


def main() -> None:
    preprocessing = joblib.load(ARTIFACTS_DIR / "preprocessing.joblib")
    encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib")
    model = joblib.load(ARTIFACTS_DIR / "random_forest.joblib")
    feature_cols = preprocessing["feature_cols"]
    raw = _load_base_data()

    # Current production evaluation: transform only with the training encoders.
    # Deliberately do not apply source-name aliases here: this must reproduce the
    # saved model's current production inputs exactly.
    feature_raw = raw.drop(columns=LEAKY_LOG_COLUMNS)
    current, _ = engineer_features(feature_raw, encoders=encoders, fit_encoders=False, save_encoders=False)
    x_train, x_test, y_train, y_test = _sample_split(current, feature_cols)
    x_test_imputed = preprocessing["imputer"].transform(x_test)
    predicted = model.predict(x_test_imputed)
    matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
    fallback_stats = {
        "prev_status_enc": _fallback_rate(raw.loc[x_test.index], encoders, "prev_status_enc", "git_prev_commit_resolution_status"),
        "lang_enc": _fallback_rate(raw.loc[x_test.index], encoders, "lang_enc", "gh_lang"),
    }

    # Requested notebook-style comparison: fit categorical encoders and imputer on
    # all data before splitting.  The aliases below are diagnostic only: they map
    # the actual TravisTorrent gh_diff_* schema to the legacy feature names used
    # by the notebook, without changing production code or artifacts.
    notebook_input = feature_raw.copy()
    notebook_input["git_diff_src_files"] = notebook_input["gh_diff_src_files"]
    notebook_input["git_diff_test_files"] = (
        notebook_input["gh_diff_tests_added"] + notebook_input["gh_diff_tests_deleted"]
    )
    notebook_like, _ = engineer_features(notebook_input, fit_encoders=True, save_encoders=False)
    nb_x_train, nb_x_test, nb_y_train, nb_y_test = _sample_split(notebook_like, feature_cols)
    full_imputer = SimpleImputer(strategy="median").fit(notebook_like[feature_cols])
    notebook_model = _rf().fit(full_imputer.transform(nb_x_train), nb_y_train)
    notebook_f1 = f1_score(nb_y_test, notebook_model.predict(full_imputer.transform(nb_x_test)))

    # Additional read-only leakage probe: adds only known post-build tr_log fields with identical RF settings.
    leaky = notebook_like.copy()
    leaky_cols = feature_cols.copy()
    for column in LEAKY_LOG_COLUMNS:
        series = raw[column]
        if pd.api.types.is_string_dtype(series) or series.dtype == "object":
            encoded_name = f"{column}_enc"
            leaky[encoded_name] = LabelEncoder().fit_transform(series.fillna("__missing__").astype(str))
            leaky_cols.append(encoded_name)
        else:
            leaky[column] = pd.to_numeric(series, errors="coerce").fillna(0.0)
            leaky_cols.append(column)
    leak_x_train, leak_x_test, leak_y_train, leak_y_test = _sample_split(leaky, leaky_cols)
    leak_imputer = SimpleImputer(strategy="median").fit(leaky[leaky_cols])
    leaky_f1 = f1_score(
        leak_y_test,
        _rf().fit(leak_imputer.transform(leak_x_train), leak_y_train).predict(leak_imputer.transform(leak_x_test)),
    )

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fallback_high = any(unseen_rate > 0.15 for _, _, _, _, unseen_rate, _, _ in fallback_stats.values())
    if fallback_high:
        verdict = "Hypothesis B confirmed — fallback encoding is high; investigate category mapping before trusting metrics."
    elif leaky_f1 >= 0.70 and notebook_f1 < 0.50:
        verdict = "Hypothesis A confirmed — 0.755 was leaked; current pipeline is correct. Full-data encoder/imputer fitting did not recover the notebook score, while post-build tr_log features did."
    else:
        verdict = "Hypothesis A confirmed — no true unseen encoder fallback signal was found, and full-dataset preprocessing does not explain the notebook score; the notebook must be audited for post-build leakage."

    print("# Random Forest Regression Diagnosis\n")
    print("## Current saved-model split")
    print(f"- Confusion matrix `[[TN, FP], [FN, TP]]`: `{matrix.tolist()}`")
    print(f"- Precision (failed): `{precision_score(y_test, predicted):.4f}`")
    print(f"- Recall (failed): `{recall_score(y_test, predicted):.4f}`")
    print(f"- F1 (failed): `{f1_score(y_test, predicted):.4f}`\n")
    print("## Encoder fallback audit")
    for feature, (count, total, rate, unseen_count, unseen_rate, missing_count, missing_rate) in fallback_stats.items():
        print(
            f"- `{feature}`: `{count}/{total}` = `{rate:.4%}` missing-or-unseen fallback; "
            f"`{unseen_count}/{total}` = `{unseen_rate:.4%}` true unseen; "
            f"`{missing_count}/{total}` = `{missing_rate:.4%}` genuine missing source value"
        )
    print("\n## Current Random Forest feature importance")
    for feature, importance in importances.items():
        print(f"- `{feature}`: `{importance:.6f}`")
    print("\n## Controlled comparison")
    print(f"- Full-dataset LabelEncoder + median imputer before split F1: `{notebook_f1:.4f}`")
    print(f"- Diagnostic model with post-build `tr_log_*` fields F1: `{leaky_f1:.4f}`")
    print("\n## Verdict")
    print(verdict)


if __name__ == "__main__":
    main()
