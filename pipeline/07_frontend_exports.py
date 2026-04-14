from __future__ import annotations

import json

import joblib
import pandas as pd

from paths import ARTIFACTS_DIR, FRONTEND_DATA_DIR, FRONTEND_DIR, ensure_runtime_dirs, stage_dir
from workflow import source_snippets


def main() -> None:
    ensure_runtime_dirs()
    training_dir = stage_dir("05_model_training")
    evaluation_dir = stage_dir("06_model_evaluation")
    stage_dir("07_frontend_exports")

    leaderboard = pd.read_csv(training_dir / "05_leaderboard.csv")
    with open(training_dir / "05_model_training_report.json", "r", encoding="utf-8") as handle:
        training_report = json.load(handle)
    with open(evaluation_dir / "06_model_evaluation_report.json", "r", encoding="utf-8") as handle:
        evaluation_report = json.load(handle)
    artifact = joblib.load(ARTIFACTS_DIR / "05_best_model.joblib")

    payload = {
        "hero": {
            "title": "Predictive Maintenance Intelligence",
            "subtitle": "Live maintenance risk monitoring, alert quality, and model diagnostics in one cinematic control room.",
        },
        "model": {
            "name": artifact["model_name"],
            "metrics": training_report["metrics"],
            "feature_count": training_report["feature_count"],
            "numeric_features": training_report["numeric_features"],
            "categorical_features": training_report["categorical_features"],
        },
        "evaluation": evaluation_report,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "workflow": {
            "stages": [
                {"id": "01", "name": "Dataset Discovery", "path": "pipeline/01_dataset_discovery.py"},
                {"id": "02", "name": "Dataset Cleaning", "path": "pipeline/02_dataset_cleaning.py"},
                {"id": "03", "name": "Training Dataset Preparation", "path": "pipeline/03_training_dataset_preparation.py"},
                {"id": "04", "name": "Feature Engineering", "path": "pipeline/04_feature_engineering.py"},
                {"id": "05", "name": "Model Training", "path": "pipeline/05_model_training.py"},
                {"id": "06", "name": "Model Evaluation", "path": "pipeline/06_model_evaluation.py"},
                {"id": "07", "name": "Frontend Exports", "path": "pipeline/07_frontend_exports.py"},
            ],
            "snippets": source_snippets(),
        },
    }

    with open(FRONTEND_DATA_DIR / "dashboard_payload.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)

    model_summary = {
        "title": payload["hero"]["title"],
        "best_model": artifact["model_name"],
        "operational_score": training_report["metrics"].get("balanced_accuracy"),
        "test_accuracy": training_report["metrics"].get("accuracy"),
        "test_roc_auc": training_report["metrics"].get("roc_auc"),
        "test_f1": training_report["metrics"].get("f1"),
        "test_average_precision": training_report["metrics"].get("average_precision"),
        "feature_count": training_report["feature_count"],
        "alert_rate": evaluation_report["operations_metrics"]["alert_rate"],
        "captured_failures": evaluation_report["operations_metrics"]["captured_failures"],
        "recommended_threshold": evaluation_report["operations_metrics"]["recommended_threshold"],
    }
    with open(FRONTEND_DIR / "model_summary.json", "w", encoding="utf-8") as handle:
        json.dump(model_summary, handle, indent=2)


if __name__ == "__main__":
    main()
