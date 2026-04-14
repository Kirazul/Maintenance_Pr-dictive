from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from paths import ARTIFACTS_DIR, LEGACY_ROOT, RAW_DATA_DIR


RAW_FILES = {
    "machine": RAW_DATA_DIR / "machine_predictive_maintenance.csv",
    "ai4i": RAW_DATA_DIR / "ai4i_2020_predictive_maintenance.csv",
}

NUMERIC_COLUMNS = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    threshold: float


def load_raw_frames() -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(path) for name, path in RAW_FILES.items()}


def _derive_failure_type_from_ai4i(df: pd.DataFrame) -> pd.Series:
    mapping = {
        "TWF": "Tool Wear Failure",
        "HDF": "Heat Dissipation Failure",
        "PWF": "Power Failure",
        "OSF": "Overstrain Failure",
        "RNF": "Random Failure",
    }
    failure_type = pd.Series("No Failure", index=df.index)
    for column, label in mapping.items():
        if column in df.columns:
            failure_type = failure_type.mask(df[column].fillna(0).astype(int) == 1, label)
    return failure_type


def canonicalize_frames(raw_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    machine = raw_frames["machine"].copy()
    machine = machine.rename(
        columns={
            "Product ID": "product_id",
            "Type": "product_type",
            "Air temperature [K]": "air_temp_k",
            "Process temperature [K]": "process_temp_k",
            "Rotational speed [rpm]": "rotational_speed_rpm",
            "Torque [Nm]": "torque_nm",
            "Tool wear [min]": "tool_wear_min",
            "Target": "machine_failure",
            "Failure Type": "failure_type",
        }
    )
    machine["udi"] = machine["UDI"]

    ai4i = raw_frames["ai4i"].copy()
    ai4i = ai4i.rename(
        columns={
            "Product ID": "product_id",
            "Type": "product_type",
            "Air temperature [K]": "air_temp_k",
            "Process temperature [K]": "process_temp_k",
            "Rotational speed [rpm]": "rotational_speed_rpm",
            "Torque [Nm]": "torque_nm",
            "Tool wear [min]": "tool_wear_min",
            "Machine failure": "machine_failure",
            "TWF": "twf",
            "HDF": "hdf",
            "PWF": "pwf",
            "OSF": "osf",
            "RNF": "rnf",
        }
    )
    ai4i["udi"] = ai4i["UDI"]
    ai4i["failure_type"] = _derive_failure_type_from_ai4i(ai4i)

    merge_keys = [
        "udi",
        "product_id",
        "product_type",
        "air_temp_k",
        "process_temp_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    ]

    machine_keep = merge_keys + ["machine_failure", "failure_type"]
    ai4i_keep = merge_keys + ["machine_failure", "failure_type", "twf", "hdf", "pwf", "osf", "rnf"]

    merged = machine[machine_keep].merge(
        ai4i[ai4i_keep],
        on=merge_keys,
        how="inner",
        suffixes=("_machine", "_ai4i"),
        validate="one_to_one",
    )

    combined = pd.DataFrame(
        {
            "udi": merged["udi"],
            "product_id": merged["product_id"],
            "product_type": merged["product_type"],
            "air_temp_k": merged["air_temp_k"],
            "process_temp_k": merged["process_temp_k"],
            "rotational_speed_rpm": merged["rotational_speed_rpm"],
            "torque_nm": merged["torque_nm"],
            "tool_wear_min": merged["tool_wear_min"],
            "machine_failure": merged["machine_failure_machine"].fillna(merged["machine_failure_ai4i"]).astype(int),
            "failure_type": merged["failure_type_machine"].fillna(merged["failure_type_ai4i"]).fillna("No Failure"),
            "source_dataset": "merged",
            "twf": merged["twf"].fillna(0).astype(int),
            "hdf": merged["hdf"].fillna(0).astype(int),
            "pwf": merged["pwf"].fillna(0).astype(int),
            "osf": merged["osf"].fillna(0).astype(int),
            "rnf": merged["rnf"].fillna(0).astype(int),
        }
    )
    combined["failure_family"] = combined["failure_type"].fillna("No Failure")
    combined["machine_failure"] = combined["machine_failure"].fillna(0).astype(int)
    return combined


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy().drop_duplicates().reset_index(drop=True)
    cleaned["product_type"] = cleaned["product_type"].fillna("unknown").astype(str)
    cleaned["failure_family"] = cleaned["failure_family"].fillna("No Failure").astype(str)
    cleaned["failure_type"] = cleaned["failure_type"].fillna("No Failure").astype(str)

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in ["twf", "hdf", "pwf", "osf", "rnf"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(0).astype(int)

    return cleaned


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["machine_failure"],
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        random_state=42,
        stratify=train_val["machine_failure"],
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features["temp_delta_k"] = features["process_temp_k"] - features["air_temp_k"]
    features["power_proxy"] = features["rotational_speed_rpm"] * features["torque_nm"]
    features["wear_power_ratio"] = features["tool_wear_min"] / (features["power_proxy"].abs() + 1.0)
    features["torque_speed_ratio"] = features["torque_nm"] / (features["rotational_speed_rpm"].abs() + 1.0)
    features["thermal_load"] = features["temp_delta_k"] * features["torque_nm"]
    features["wear_temp_stress"] = features["tool_wear_min"] * features["temp_delta_k"]
    features["high_torque_flag"] = (features["torque_nm"] >= 55).astype(int)
    features["low_speed_flag"] = (features["rotational_speed_rpm"] <= 1400).astype(int)
    features["wear_risk_band"] = pd.cut(
        features["tool_wear_min"],
        bins=[-np.inf, 50, 120, 180, np.inf],
        labels=["fresh", "moderate", "elevated", "critical"],
    ).astype(str)
    return features


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [
        "air_temp_k",
        "process_temp_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "temp_delta_k",
        "power_proxy",
        "wear_power_ratio",
        "torque_speed_ratio",
        "thermal_load",
        "wear_temp_stress",
        "high_torque_flag",
        "low_speed_flag",
    ]
    categorical = ["product_type", "source_dataset", "wear_risk_band"]
    return numeric, categorical


def build_preprocessor(numeric: Iterable[str], categorical: Iterable[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                list(numeric),
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                list(categorical),
            ),
        ]
    )


def evaluate_classifier(model: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict[str, float]:
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
    }


def select_operating_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.arange(0.1, 0.91, 0.05):
        predictions = (probabilities >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, predictions)
        if score > best_score:
            best_score = score
            best_threshold = float(round(threshold, 2))
    return best_threshold


def train_model_suite(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[list[ModelResult], dict]:
    numeric, categorical = feature_columns(train_df)
    X_train = train_df[numeric + categorical]
    y_train = train_df["machine_failure"]
    X_validation = validation_df[numeric + categorical]
    y_validation = validation_df["machine_failure"]
    X_test = test_df[numeric + categorical]
    y_test = test_df["machine_failure"]

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    results: list[ModelResult] = []
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(numeric, categorical)),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        validation_probabilities = pipeline.predict_proba(X_validation)[:, 1]
        threshold = select_operating_threshold(y_validation, validation_probabilities)
        validation_metrics = evaluate_classifier(pipeline, X_validation, y_validation, threshold)
        test_metrics = evaluate_classifier(pipeline, X_test, y_test, threshold)
        results.append(ModelResult(name=name, pipeline=pipeline, validation_metrics=validation_metrics, test_metrics=test_metrics, threshold=threshold))

    results.sort(key=lambda item: (item.validation_metrics["balanced_accuracy"], item.validation_metrics["roc_auc"]), reverse=True)
    best = results[0]

    combined_train = pd.concat([train_df, validation_df], ignore_index=True)
    X_combined = combined_train[numeric + categorical]
    y_combined = combined_train["machine_failure"]
    best.pipeline.fit(X_combined, y_combined)

    probabilities = best.pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= best.threshold).astype(int)
    confusion = confusion_matrix(y_test, predictions)
    artifact = {
        "model_name": best.name,
        "pipeline": best.pipeline,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "threshold": best.threshold,
        "test_metrics": evaluate_classifier(best.pipeline, X_test, y_test, best.threshold),
        "test_probabilities": probabilities,
        "test_predictions": predictions,
        "test_target": y_test.to_numpy(),
        "confusion_matrix": confusion.tolist(),
    }
    joblib.dump(artifact, ARTIFACTS_DIR / "05_best_model.joblib")
    return results, artifact


def threshold_report(y_true: np.ndarray, probabilities: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for threshold in np.arange(0.1, 0.95, 0.05):
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(round(threshold, 2)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
                "precision": float(precision_score(y_true, predictions, zero_division=0)),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
                "f1": float(f1_score(y_true, predictions, zero_division=0)),
                "alert_rate": float(np.mean(predictions)),
            }
        )
    return rows


def write_json(path: Path, payload: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def source_snippets() -> list[dict[str, str]]:
    snippets = []
    for relative in [
        "pipeline/02_dataset_cleaning.py",
        "pipeline/04_feature_engineering.py",
        "pipeline/05_model_training.py",
        "api/app.py",
    ]:
        path = LEGACY_ROOT.parent / "maintenance" / relative
        if path.exists():
            content = path.read_text(encoding="utf-8")
            snippets.append({"path": relative, "content": "\n".join(content.splitlines()[:60])})
    return snippets
