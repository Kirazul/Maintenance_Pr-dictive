from __future__ import annotations

import pandas as pd
from workflow import canonicalize_frames, clean_dataset, load_raw_frames, write_json
from paths import ensure_runtime_dirs, stage_dir


def fix_inconsistent_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix known labeling inconsistencies in the dataset.
    
    Issues identified:
    1. "Random Failures" with machine_failure=0
    2. "No Failure" with machine_failure=1
    3. Mismatches between failure_type and individual flags
    """
    df = df.copy()
    
    # Issue 1: Random Failures should have machine_failure=1
    random_failures = (df['failure_type'] == 'Random Failures') & (df['machine_failure'] == 0)
    if random_failures.sum() > 0:
        print(f"  Fixing {random_failures.sum()} 'Random Failures' with machine_failure=0")
        df.loc[random_failures, 'machine_failure'] = 1
    
    # Issue 2: Investigate "No Failure" with machine_failure=1
    no_failure_but_failure = (df['failure_type'] == 'No Failure') & (df['machine_failure'] == 1)
    if no_failure_but_failure.sum() > 0:
        print(f"  Investigating {no_failure_but_failure.sum()} records with 'No Failure' but machine_failure=1")
        # These might be data entry errors - set to machine_failure=0
        # Alternatively, could infer failure type from flags
        df.loc[no_failure_but_failure, 'machine_failure'] = 0
        print(f"    Set machine_failure=0 for these records")
    
    # Issue 3: Ensure consistency between failure_type and failure_family
    df['failure_family'] = df['failure_type'].fillna('No Failure')
    
    # Verify flag consistency for specific failure types
    flag_mapping = {
        'Tool Wear Failure': 'twf',
        'Heat Dissipation Failure': 'hdf',
        'Power Failure': 'pwf',
        'Overstrain Failure': 'osf',
        'Random Failures': 'rnf'
    }
    
    # For records with specific failure types, ensure corresponding flag is set
    for failure_type, flag_col in flag_mapping.items():
        mask = df['failure_type'] == failure_type
        if mask.sum() > 0:
            missing_flags = mask & (df[flag_col] == 0)
            if missing_flags.sum() > 0:
                print(f"  Setting {flag_col}=1 for {missing_flags.sum()} '{failure_type}' records")
                df.loc[missing_flags, flag_col] = 1
    
    return df


def main() -> None:
    ensure_runtime_dirs()
    output_dir = stage_dir("02_cleaning")
    
    print("Loading raw datasets...")
    raw_frames = load_raw_frames()
    
    print("Canonicalizing frames...")
    canonical = canonicalize_frames(raw_frames)
    
    print("Cleaning dataset...")
    cleaned_df = clean_dataset(canonical)
    
    print("Fixing inconsistent labels...")
    cleaned_df = fix_inconsistent_labels(cleaned_df)
    
    # Save cleaned dataset
    cleaned_path = output_dir / "02_cleaned_dataset.csv"
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned dataset to {cleaned_path}")
    
    # Check for remaining inconsistencies
    inconsistent = (
        ((cleaned_df['failure_type'] == 'Random Failures') & (cleaned_df['machine_failure'] == 0)) |
        ((cleaned_df['failure_type'] == 'No Failure') & (cleaned_df['machine_failure'] == 1))
    ).sum()
    
    report = {
        "rows": int(cleaned_df.shape[0]),
        "columns": int(cleaned_df.shape[1]),
        "target_positive_rate": float(cleaned_df["machine_failure"].mean()),
        "source_mix": cleaned_df["source_dataset"].value_counts().to_dict(),
        "failure_mix": cleaned_df["failure_family"].value_counts().head(10).to_dict(),
        "null_counts": cleaned_df.isna().sum().sort_values(ascending=False).head(10).to_dict(),
        "remaining_inconsistencies": int(inconsistent),
    }
    write_json(output_dir / "02_cleaning_report.json", report)
    print("Label cleaning complete!")


if __name__ == "__main__":
    main()
