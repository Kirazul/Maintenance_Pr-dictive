# Pipeline

This folder contains the active numbered workflow for the `maintenance/` product app.

## Execution Order

1. `01_dataset_discovery.py`
2. `02_dataset_cleaning.py`
3. `03_training_dataset_preparation.py`
4. `04_feature_engineering.py`
5. `05_model_training.py`
6. `06_model_evaluation.py`
7. `07_frontend_exports.py`
8. `08_run_pipeline.py`

## Inputs

- `../Maintenance_Pr-dictive_4.1-main/data/machine_predictive_maintenance.csv`
- `../Maintenance_Pr-dictive_4.1-main/data/ai4i_2020_predictive_maintenance.csv`

## Outputs

- stage-organized artifacts under `data/processed/01_discovery/` through `data/processed/07_frontend_exports/`
- trained model artifact under `artifacts/05_best_model.joblib`
- frontend summary under `frontend/model_summary.json`
- frontend dashboard data under `frontend/assets/data/dashboard_payload.json`
