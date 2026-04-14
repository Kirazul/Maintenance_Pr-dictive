from __future__ import annotations

import pandas as pd

from paths import ensure_runtime_dirs, stage_dir
from workflow import split_dataset, write_json


def main() -> None:
    ensure_runtime_dirs()
    input_dir = stage_dir("02_cleaning")
    output_dir = stage_dir("03_training_dataset")
    cleaned_df = pd.read_csv(input_dir / "02_cleaned_dataset.csv")
    train_df, validation_df, test_df = split_dataset(cleaned_df)

    train_df.to_csv(output_dir / "train_data.csv", index=False)
    validation_df.to_csv(output_dir / "validation_data.csv", index=False)
    test_df.to_csv(output_dir / "test_data.csv", index=False)

    report = {
        "training_rows": int(train_df.shape[0]),
        "validation_rows": int(validation_df.shape[0]),
        "test_rows": int(test_df.shape[0]),
        "train_positive_rate": float(train_df["machine_failure"].mean()),
        "validation_positive_rate": float(validation_df["machine_failure"].mean()),
        "test_positive_rate": float(test_df["machine_failure"].mean()),
    }
    write_json(output_dir / "03_training_dataset_report.json", report)


if __name__ == "__main__":
    main()
