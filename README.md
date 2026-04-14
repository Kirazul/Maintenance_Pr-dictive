# Predictive Maintenance Intelligence System

A complete end-to-end machine learning workflow for predictive maintenance. The system combines dataset reconciliation, calibrated failure classification, threshold-aware evaluation, a FastAPI prediction service, and a cinematic frontend control room inspired by `RealEstate_V2`.

---

## Problem Statement

Predictive maintenance projects often look better on paper than they are in production. This repository addresses that problem directly: it rebuilds the original maintenance work into a productized workflow that removes duplicate-row leakage, avoids target leakage, calibrates alert thresholds, and presents operational metrics honestly.

---

## Architecture Overview

The system consists of three interconnected layers:

| Layer | Description |
|-------|-------------|
| **Data Pipeline** | Eight-stage workflow from raw machine records to calibrated classifier artifacts |
| **Prediction API** | FastAPI service exposing maintenance diagnostics, notebook execution, and live predictions |
| **Control Room + ML Lab** | Cinematic frontend dashboard plus interactive workflow notebook |

All layers share common exported artifacts so training, diagnostics, and serving remain aligned.

---

## Pipeline Stages

### Stage 1: Dataset Discovery

Location: `pipeline/01_dataset_discovery.py`

> Profiles the raw maintenance CSV files and documents schema, row counts, and basic data quality characteristics for each source.

### Stage 2: Dataset Cleaning

Location: `pipeline/02_dataset_cleaning.py`

> Canonicalizes both raw datasets into one schema, merges them into one record per machine, standardizes labels, and removes duplicate-row inflation from the training path.

### Stage 3: Training Dataset Preparation

Location: `pipeline/03_training_dataset_preparation.py`

> Builds stratified train, validation, and test splits using the unified machine failure target.

### Stage 4: Feature Engineering

Location: `pipeline/04_feature_engineering.py`

> Creates interpretable operational features including thermal spread, power proxy, torque-speed ratio, wear stress, and categorical wear risk bands.

### Stage 5: Model Training

Location: `pipeline/05_model_training.py`

> Benchmarks multiple classifiers, selects the operating threshold on validation data, and exports the best production artifact for serving.

### Stage 6: Model Evaluation

Location: `pipeline/06_model_evaluation.py`

> Reports threshold-aware holdout metrics including balanced accuracy, precision, recall, F1, ROC AUC, average precision, and confusion matrix behavior.

### Stage 7: Frontend Exports

Location: `pipeline/07_frontend_exports.py`

> Writes dashboard payloads, model summary exports, workflow snippets, and notebook-facing data used by the UI.

### Stage 8: Full Pipeline Runner

Location: `pipeline/08_run_pipeline.py`

> Executes the full workflow end to end in the correct stage order.

---

## Evaluation Philosophy

The target is highly imbalanced, with machine failures representing a small fraction of records. Because of that, plain accuracy is not the primary production score.

The project emphasizes:

- **Balanced Accuracy** as the headline operational score
- **Recall** to capture true failure events
- **Precision** to control false maintenance alerts
- **Threshold tuning** on validation data before holdout evaluation

This makes the system more honest and more deployable than a naive high-accuracy classifier on an imbalanced dataset.

---

## Current Model Performance

Current exported production summary:

- **Algorithm**: RandomForest Classifier
- **Operational Score (Balanced Accuracy)**: 0.930
- **Raw Accuracy**: 0.961
- **Test ROC AUC**: 0.978
- **Test F1**: 0.610
- **Average Precision**: 0.843
- **Recommended Threshold**: 0.10
- **Failure Capture Rate**: 89.71%

---

## API Endpoints

The FastAPI application provides:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict machine failure probability for a scenario |
| `/model_summary` | GET | Retrieve frontend-ready model summary metrics |
| `/dashboard` | GET | Retrieve full dashboard payload |
| `/sample_observations` | GET | Return sample holdout observations |
| `/api/source` | GET | Return project source file contents |
| `/api/source/raw` | GET | Download/view a raw source file |
| `/api/source/rendered` | GET | Render notebook files as HTML |
| `/run-cell` | POST | Execute notebook cells through the API |

---

## Frontend Components

### Control Room

Location: `frontend/index.html`

The main interface presents the predictive maintenance system as a cinematic command deck. It includes:

- calibrated operating score and alert policy indicators
- scenario simulation inputs for machine state
- model competition leaderboard
- threshold sweep diagnostics
- workflow stage visibility

### ML Lab Notebook

Location: `frontend/notebook.html`

The notebook follows the same product pattern as `RealEstate_V2`: dashboard diagnostics, chaptered workflow cards, runnable code cells, source browsing, and raw notebook rendering.

---

## Output Artifacts

### Model Artifact

- `artifacts/05_best_model.joblib`

### Processed Data

```text
data/processed/
├── 01_discovery/
├── 02_cleaning/
├── 03_training_dataset/
├── 04_feature_engineering/
├── 05_model_training/
├── 06_model_evaluation/
└── 07_frontend_exports/
```

### Frontend Exports

```text
frontend/assets/data/
└── dashboard_payload.json

frontend/model_summary.json
```

---

## Quick Start

### Run the Full Pipeline

```bash
py -3.11 pipeline/08_run_pipeline.py
```

### Run the API

```bash
py -3.11 -m uvicorn api.app:app --reload --port 8000
```

Access API docs at `http://127.0.0.1:8000/docs`

### Open the Product UI

- Control Room: `http://127.0.0.1:8000/`
- ML Lab: `http://127.0.0.1:8000/notebook`

---

## Project Layout

```text
maintenance/
├── api/
├── artifacts/
├── data/
│   └── processed/
├── frontend/
│   ├── assets/
│   ├── css/
│   └── js/
├── pipeline/
├── Maintenance_Complete_Pipeline.ipynb
├── Procfile
├── render.yaml
└── run_server.bat
```

---

## Notes

- The product structure is intentionally aligned with `RealEstate_V2`.
- The maintenance workflow is stricter about leakage and threshold calibration.
- The notebook layer is now interactive and backed by API-executed cells rather than static summaries alone.
