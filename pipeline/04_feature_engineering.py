from __future__ import annotations

import pandas as pd

from paths import ensure_runtime_dirs, stage_dir
from workflow import engineer_features, feature_columns, write_json


def main() -> None:
    ensure_runtime_dirs()
    input_dir = stage_dir("03_training_dataset")
    output_dir = stage_dir("04_feature_engineering")

    datasets = {
        "train": pd.read_csv(input_dir / "train_data.csv"),
        "validation": pd.read_csv(input_dir / "validation_data.csv"),
        "test": pd.read_csv(input_dir / "test_data.csv"),
    }
    engineered = {name: engineer_features(frame) for name, frame in datasets.items()}

    for name, frame in engineered.items():
        frame.to_csv(output_dir / f"{name}_features.csv", index=False)

    numeric, categorical = feature_columns(engineered["train"])
    feature_snapshot = {
        "training_rows": int(engineered["train"].shape[0]),
        "numeric_feature_count": len(numeric),
        "categorical_feature_count": len(categorical),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "positive_rate": float(engineered["train"]["machine_failure"].mean()),
    }
    write_json(output_dir / "04_feature_engineering_report.json", feature_snapshot)


if __name__ == "__main__":
    main()
