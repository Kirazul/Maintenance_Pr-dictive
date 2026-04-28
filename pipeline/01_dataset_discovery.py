from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
from paths import RAW_DATA_DIR, ensure_runtime_dirs, stage_dir


INPUT_FILES = [
    "machine_predictive_maintenance.csv",
    "ai4i_2020_predictive_maintenance.csv",
]


def main() -> None:
    ensure_runtime_dirs()
    output_dir = stage_dir("01_discovery")

    manifest = {"datasets": []}
    for filename in INPUT_FILES:
        path = RAW_DATA_DIR / filename
        if not path.exists():
            print(f"Warning: {filename} not found in {RAW_DATA_DIR}")
            continue
            
        df = pd.read_csv(path)
        profile = {
            "columns": list(df.columns),
            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
            "null_counts": df.isna().sum().to_dict(),
            "sample": df.head(5).to_dict(orient="records"),
            "stats": df.describe().to_dict()
        }
        
        output_path = output_dir / f"{filename.replace('.csv', '')}_profile.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(profile, handle, indent=2, default=str)
        manifest["datasets"].append(
            {
                "file": filename,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "profile": str(output_path.name),
            }
        )

    with open(output_dir / "01_discovery_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
