from __future__ import annotations

from workflow import canonicalize_frames, clean_dataset, load_raw_frames, write_json
from paths import ensure_runtime_dirs, stage_dir


def main() -> None:
    ensure_runtime_dirs()
    output_dir = stage_dir("02_cleaning")
    raw_frames = load_raw_frames()
    canonical = canonicalize_frames(raw_frames)
    cleaned_df = clean_dataset(canonical)
    cleaned_path = output_dir / "02_cleaned_dataset.csv"
    cleaned_df.to_csv(cleaned_path, index=False)

    report = {
        "rows": int(cleaned_df.shape[0]),
        "columns": int(cleaned_df.shape[1]),
        "target_positive_rate": float(cleaned_df["machine_failure"].mean()),
        "source_mix": cleaned_df["source_dataset"].value_counts().to_dict(),
        "failure_mix": cleaned_df["failure_family"].value_counts().head(10).to_dict(),
        "null_counts": cleaned_df.isna().sum().sort_values(ascending=False).head(10).to_dict(),
    }
    write_json(output_dir / "02_cleaning_report.json", report)


if __name__ == "__main__":
    main()
