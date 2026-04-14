from __future__ import annotations

import pandas as pd

from paths import ensure_runtime_dirs, stage_dir
from workflow import train_model_suite, write_json


def main() -> None:
    ensure_runtime_dirs()
    data_dir = stage_dir("04_feature_engineering")
    output_dir = stage_dir("05_model_training")

    train_df = pd.read_csv(data_dir / "train_features.csv")
    validation_df = pd.read_csv(data_dir / "validation_features.csv")
    test_df = pd.read_csv(data_dir / "test_features.csv")

    results, artifact = train_model_suite(train_df, validation_df, test_df)
    leaderboard = pd.DataFrame(
        [
            {
                "model_name": result.name,
                "threshold": result.threshold,
                "validation_accuracy": result.validation_metrics["accuracy"],
                "validation_balanced_accuracy": result.validation_metrics["balanced_accuracy"],
                "validation_precision": result.validation_metrics["precision"],
                "validation_recall": result.validation_metrics["recall"],
                "validation_f1": result.validation_metrics["f1"],
                "validation_roc_auc": result.validation_metrics["roc_auc"],
                "validation_average_precision": result.validation_metrics["average_precision"],
                "test_accuracy": result.test_metrics["accuracy"],
                "test_balanced_accuracy": result.test_metrics["balanced_accuracy"],
                "test_precision": result.test_metrics["precision"],
                "test_recall": result.test_metrics["recall"],
                "test_f1": result.test_metrics["f1"],
                "test_roc_auc": result.test_metrics["roc_auc"],
                "test_average_precision": result.test_metrics["average_precision"],
            }
            for result in results
        ]
    )
    leaderboard.to_csv(output_dir / "05_leaderboard.csv", index=False)

    report = {
        "best_model": artifact["model_name"],
        "metrics": artifact["test_metrics"],
        "feature_count": len(artifact["numeric_features"]) + len(artifact["categorical_features"]),
        "numeric_features": artifact["numeric_features"],
        "categorical_features": artifact["categorical_features"],
        "threshold": artifact["threshold"],
        "headline_metric": "balanced_accuracy",
        "leaderboard_preview": leaderboard.to_dict(orient="records"),
    }
    write_json(output_dir / "05_model_training_report.json", report)


if __name__ == "__main__":
    main()
