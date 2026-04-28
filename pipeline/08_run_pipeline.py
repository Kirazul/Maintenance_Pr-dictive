from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
STEPS = [
    "01_dataset_discovery.py",
    "02_dataset_cleaning.py",
    "03_training_dataset_preparation.py",
    "04_feature_engineering.py",
    "05_model_training.py",
    "06_model_evaluation.py",
    "07_frontend_exports.py",
]


def main() -> None:
    for step in STEPS:
        print(f"\n=== Running {step} ===")
        subprocess.run([sys.executable, str(PIPELINE_DIR / step)], check=True)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
