from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str((Path(__file__).resolve().parents[1].parent / "Maintenance_Pr-dictive_4.1-main" / "src").resolve()))

from etl.data_loader import DataLoader

from paths import RAW_DATA_DIR, ensure_runtime_dirs, stage_dir


INPUT_FILES = [
    "machine_predictive_maintenance.csv",
    "ai4i_2020_predictive_maintenance.csv",
]


def main() -> None:
    ensure_runtime_dirs()
    output_dir = stage_dir("01_discovery")
    loader = DataLoader(str((RAW_DATA_DIR.parent / "config.yaml").resolve()))
    loader.data_dir = RAW_DATA_DIR

    manifest = {"datasets": []}
    for filename in INPUT_FILES:
        df = loader.load_csv(filename)
        profile = loader.get_data_profile(df)
        output_path = output_dir / f"{filename.replace('.csv', '')}_profile.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(profile, handle, indent=2, default=str)
        manifest["datasets"].append(
            {
                "file": filename,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "profile": str(output_path.relative_to(output_dir.parent.parent.parent)),
            }
        )

    with open(output_dir / "01_discovery_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
