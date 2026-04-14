from __future__ import annotations

import joblib

from paths import ARTIFACTS_DIR, ensure_runtime_dirs, stage_dir
from workflow import threshold_report, write_json


def main() -> None:
    ensure_runtime_dirs()
    output_dir = stage_dir("06_model_evaluation")
    artifact = joblib.load(ARTIFACTS_DIR / "05_best_model.joblib")
    y_true = artifact["test_target"]
    y_pred = artifact["test_predictions"]
    probabilities = artifact["test_probabilities"]

    tn, fp, fn, tp = artifact["confusion_matrix"][0][0], artifact["confusion_matrix"][0][1], artifact["confusion_matrix"][1][0], artifact["confusion_matrix"][1][1]
    threshold_curve = threshold_report(y_true, probabilities)
    best_threshold = max(threshold_curve, key=lambda row: row["balanced_accuracy"])

    report = {
        "model_name": artifact["model_name"],
        "classification_metrics": artifact["test_metrics"],
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "operations_metrics": {
            "alert_rate": float((y_pred == 1).mean()),
            "failure_prevalence": float(y_true.mean()),
            "captured_failures": float(tp / max(tp + fn, 1)),
            "false_alarm_share": float(fp / max(tp + fp, 1)),
            "operational_score": artifact["test_metrics"]["balanced_accuracy"],
            "recommended_threshold": best_threshold["threshold"],
        },
        "threshold_curve": threshold_curve,
    }

    write_json(output_dir / "06_model_evaluation_report.json", report)


if __name__ == "__main__":
    main()
