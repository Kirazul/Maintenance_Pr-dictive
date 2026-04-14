from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = PROJECT_ROOT.parent / "Maintenance_Pr-dictive_4.1-main"

RAW_DATA_DIR = LEGACY_ROOT / "data"
LEGACY_RESULTS_DIR = LEGACY_ROOT / "results"

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_DATA_DIR = FRONTEND_DIR / "assets" / "data"


def stage_dir(stage: str) -> Path:
    directory = PROCESSED_DIR / stage
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_runtime_dirs() -> None:
    for directory in [
        DATA_DIR,
        PROCESSED_DIR,
        ARTIFACTS_DIR,
        FRONTEND_DIR,
        FRONTEND_DIR / "assets",
        FRONTEND_DIR / "assets" / "icons",
        FRONTEND_DATA_DIR,
        FRONTEND_DIR / "css",
        FRONTEND_DIR / "js",
    ]:
        directory.mkdir(parents=True, exist_ok=True)
