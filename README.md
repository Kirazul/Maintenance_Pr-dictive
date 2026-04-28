# Predictive Maintenance Intelligence System

A complete end-to-end machine learning system for predicting machine failures in industrial manufacturing environments. The system handles severe class imbalance (28.6:1 ratio), provides calibrated risk assessments, and includes an interactive presentation with scenario simulation.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Status](https://img.shields.io/badge/Status-Production_Ready-green)
![Accuracy](https://img.shields.io/badge/Balanced_Accuracy-90.89%25-success)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Project Architecture](#project-architecture)
3. [Key Features](#key-features)
4. [Data Overview](#data-overview)
5. [Model Performance](#model-performance)
6. [Frontend Interface](#frontend-interface)
7. [API Endpoints](#api-endpoints)
8. [Project Structure](#project-structure)
9. [Quick Start](#quick-start)
10. [Pipeline Details](#pipeline-details)
11. [Technical Notes](#technical-notes)

---

## Problem Statement

Predictive maintenance aims to prevent unexpected machine failures that can cause costly production downtime. However, building effective predictive models faces significant challenges:

- **Severe Class Imbalance**: Only ~3.4% of records represent actual failures (28.6:1 ratio)
- **Misleading Accuracy**: Naive models achieve ~98% accuracy by simply predicting "no failure" for everything
- **Data Quality Issues**: Duplicate rows, target leakage, and inconsistent schemas across datasets
- **Threshold Calibration**: Default 50% threshold is inappropriate for high-stakes failure prediction

This project addresses all these challenges with a production-ready pipeline that emphasizes **balanced accuracy**, **recall**, and **threshold tuning** over raw accuracy.

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND LAYER                               │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐                │
│  │Control Room │  │  ML Lab     │  │ Presentation │                │
│  │ (Dashboard) │  │ (Notebook)  │  │ (13 Slides)  │                │
│  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘                │
└─────────┼────────────────┼────────────────┼─────────────────────────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
                    ┌──────▼──────┐
                    │  FastAPI    │
                    │   Server    │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
    │ Predict   │   │ Dashboard  │   │  Notebook │
    │  Service  │   │   Data     │   │  Runner   │
    └───────────┘   └───────────┘   └───────────┘
                           │
                    ┌──────▼──────┐
                    │ML PIPELINE  │
                    └─────────────┘
```

### Three-Layer System

| Layer | Description |
|-------|-------------|
| **Frontend** | Cinematic control room dashboard, interactive ML lab notebook, and 13-slide presentation with scenario simulator |
| **API** | FastAPI service providing predictions, dashboard data, and notebook execution |
| **Pipeline** | 8-stage ML workflow from raw data to trained model with full validation |

---

## Key Features

### 🔧 ML Pipeline Features

- **8-Stage Pipeline**: Dataset discovery → Cleaning → Feature engineering → Model training → Evaluation → Export
- **Multiple Models**: RandomForest, GradientBoosting, LogisticRegression, SVM, XGBoost comparison
- **Feature Engineering**: 16 engineered features including power_proxy, torque_speed_ratio, thermal_load, wear_temp_stress
- **Threshold Calibration**: Automatic threshold tuning on validation data (optimal: 0.25)
- **Class Imbalance Handling**: Stratified splits, focus on balanced metrics

### 🎨 Frontend Features

- **Control Room Dashboard**: Real-time failure probability gauge, scenario simulator, model leaderboard
- **ML Lab Notebook**: Interactive workflow with runnable code cells, chaptered navigation
- **Presentation Mode**: 13-slide cinematic presentation with scenario simulator
- **Glassmorphism UI**: Modern translucent design with blur effects

### 📊 Presentation Slides

1. Title & Introduction
2. Problem Statement
3. Data Sources (AI4I + Machine datasets)
4. Class Imbalance Analysis
5. Feature Engineering Overview
6. Model Comparison Matrix
7. Feature Importance Analysis
8. Critical Thresholds
9. Model Performance Metrics
10. Threshold Sweep Visualization
11. **Scenario Simulator** (Interactive)
12. Next Steps
13. Q&A

---

## Data Overview

### Source Datasets

| Dataset | Records | Description |
|---------|---------|--------------|
| `ai4i_2020_predictive_maintenance.csv` | 10,000 | AI4I 2020 benchmark dataset |
| `machine_predictive_maintenance.csv` | 7,829 | Additional machine records |

### Combined Dataset

- **Total Records**: ~17,800 (after deduplication)
- **Features**: Type, Air Temp, Process Temp, Rotational Speed, Torque, Tool Wear
- **Target**: Machine Failure (binary: 0/1)

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| No Failure | ~17,200 | 96.6% |
| Failure | ~600 | 3.4% |
| **Imbalance Ratio** | - | **28.6:1** |

### Key Predictive Features

| Feature | Importance | Description |
|---------|------------|--------------|
| power_proxy | 29.6% | Calculated power consumption |
| rotational_speed_rpm | 26.6% | Machine RPM |
| tool_wear_min | 16.0% | Tool usage time |
| temp_delta_k | 10.8% | Temperature difference |
| torque_nm | 5.7% | Torque measurement |

### Critical Thresholds

| Condition | Failure Rate | Risk Level |
|-----------|--------------|-------------|
| Torque ≥ 55 Nm | 24.3% | **Critical** |
| Speed ≤ 1400 rpm | 12.6% | **High** |
| Tool Wear > 180 min | 8.4% | **Warning** |

---

## Model Performance

### Best Model: Gradient Boosting Classifier

| Metric | Score | Notes |
|--------|-------|-------|
| **Balanced Accuracy** | 90.89% | Primary operational metric |
| Accuracy | 98.85% | Misleading due to imbalance |
| Precision | 83.58% | Controls false alarms |
| Recall | 82.35% | Captures actual failures |
| F1 Score | 82.96% | Harmonic mean |
| ROC AUC | 97.90% | Discrimination ability |
| Average Precision | 85.90% | PR curve area |

### Failure Capture

- **Captured Failures**: 82.35%
- **Alert Rate**: 3.8% of predictions trigger alerts
- **Recommended Threshold**: 0.25

### Risk Band Classification

| Probability Range | Risk Band | Action |
|------------------|-----------|--------|
| < 17.5% | 🟢 Stable | Continue monitoring |
| 17.5% - 65% | 🟡 Warning | Schedule inspection |
| ≥ 65% | 🔴 Critical | Inspect immediately |

---

## Frontend Interface

### Control Room (index.html)

The main dashboard presents:
- Failure probability gauge with animated signal
- Scenario simulator with live prediction
- Model leaderboard showing algorithm comparisons
- Threshold sweep visualization
- Quick action buttons (Stable/Warning/Critical presets)

### ML Lab Notebook (notebook.html)

Interactive notebook with:
- Chapter-based workflow cards
- Runnable code cells via API
- Source file browser
- Inline execution results

### Presentation Mode (presentation.html)

13-slide cinematic presentation:
- Keyboard navigation (arrow keys)
- Interactive scenario simulator on slide 11
- Glassmorphism navigation buttons
- Progress indicator

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Control Room dashboard |
| `/notebook` | GET | ML Lab notebook interface |
| `/presentation` | GET | 13-slide presentation |
| `/predict` | POST | Predict failure probability |
| `/model_summary` | GET | Model metrics and thresholds |
| `/dashboard` | GET | Full dashboard payload |
| `/sample_observations` | GET | Sample holdout data |
| `/run-cell` | POST | Execute notebook code |

---

## Project Structure

```
Maintenance_Pr-dictive/
├── api/
│   └── app.py              # FastAPI server
├── artifacts/             # Model artifacts (generated)
├── data/
│   ├── ai4i_2020_predictive_maintenance.csv
│   ├── machine_predictive_maintenance.csv
│   └── processed/         # Pipeline outputs
├── frontend/
│   ├── index.html         # Control Room
│   ├── notebook.html      # ML Lab
│   ├── presentation.html  # 13-slide presentation
│   ├── model_summary.json # Model metrics
│   ├── css/style.css
│   ├── js/app.js
│   ├── js/notebook.js
│   └── assets/data/dashboard_payload.json
├── pipeline/
│   ├── 01_dataset_discovery.py
│   ├── 02_dataset_cleaning.py
│   ├── 03_training_dataset_preparation.py
│   ├── 04_feature_engineering.py
│   ├── 05_model_training.py
│   ├── 06_model_evaluation.py
│   ├── 07_frontend_exports.py
│   ├── 08_run_pipeline.py
│   ├── workflow.py
│   └── paths.py
├── run_pipeline.py        # Run full pipeline
├── README.md
└── requirements.txt       # Dependencies
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or use:
```bash
pip install pandas numpy scikit-learn joblib fastapi uvicorn
```

### 2. Run Full Pipeline

```bash
python run_pipeline.py
```

Or:
```bash
python pipeline/08_run_pipeline.py
```

This executes all 8 pipeline stages:
1. Dataset Discovery
2. Dataset Cleaning
3. Training Dataset Preparation
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Frontend Exports
8. Full Pipeline Runner

### 3. Start API Server

```bash
uvicorn api.app:app --reload --port 8000
```

Or on Windows:
```bash
py -3.11 -m uvicorn api.app:app --reload --port 8000
```

### 4. Open Frontend

| Interface | URL |
|-----------|-----|
| Control Room | http://127.0.0.1:8000/ |
| ML Lab Notebook | http://127.0.0.1:8000/notebook |
| Presentation | http://127.0.0.1:8000/presentation |
| API Docs | http://127.0.0.1:8000/docs |

---

## Pipeline Details

### Stage 1: Dataset Discovery

**File**: `pipeline/01_dataset_discovery.py`

- Profiles raw CSV files
- Documents schema, row counts, data quality
- Identifies feature types and distributions

### Stage 2: Dataset Cleaning

**File**: `pipeline/02_dataset_cleaning.py`

- Canonicalizes schemas across datasets
- Merges into unified format
- Removes duplicate rows
- Standardizes target variable

### Stage 3: Training Dataset Preparation

**File**: `pipeline/03_training_dataset_preparation.py`

- Stratified train/validation/test splits
- Maintains class distribution
- 70/15/15 split ratio

### Stage 4: Feature Engineering

**File**: `pipeline/04_feature_engineering.py`

Creates 16 features:
- `temp_delta_k`: Process temp - Air temp
- `power_proxy`: Torque × Speed
- `wear_power_ratio`: Tool wear / Power proxy
- `torque_speed_ratio`: Torque / Speed
- `thermal_load`: temp_delta_k × power_proxy
- `wear_temp_stress`: tool_wear × temp_delta_k
- `high_torque_flag`: Torque ≥ 55 Nm
- `low_speed_flag`: Speed ≤ 1400 rpm
- `wear_risk_band`: Categorical (Low/Medium/High/Critical)

### Stage 5: Model Training

**File**: `pipeline/05_model_training.py`

Benchmarks algorithms:
- RandomForest
- GradientBoosting
- LogisticRegression
- SVM
- XGBoost

Selects best based on balanced accuracy

### Stage 6: Model Evaluation

**File**: `pipeline/06_model_evaluation.py`

Reports metrics:
- Balanced accuracy
- Precision/Recall/F1
- ROC AUC
- Average Precision
- Confusion matrix
- Threshold sweep

### Stage 7: Frontend Exports

**File**: `pipeline/07_frontend_exports.py`

Writes:
- `frontend/model_summary.json`
- `frontend/assets/data/dashboard_payload.json`
- Workflow snippets for notebook

### Stage 8: Full Pipeline Runner

**File**: `pipeline/08_run_pipeline.py`

Executes all stages in correct order

---

## Technical Notes

### Class Imbalance Handling

The system explicitly handles the 28.6:1 class imbalance:
- Uses balanced accuracy as primary metric
- Calibrates threshold to 0.25 (not 0.50)
- Tracks failure capture rate separately

### Why Not Plain Accuracy?

With 96.6% non-failures, a naive model predicting "no failure" for everything achieves ~97% accuracy. This is misleading for maintenance operations where **detecting actual failures** (recall) is critical.

### Prediction API

The `/predict` endpoint:
1. Loads training data
2. Trains model inline (avoids pickle compatibility issues)
3. Generates probability for input features
4. Returns risk band and recommended action

### Presentation Simulator

Slide 11 includes interactive controls:
- Air Temperature, Process Temperature
- Rotational Speed, Torque, Tool Wear
- Product Type selection
- Preset buttons (Stable/Warning/Critical)
- Live prediction updates

---

## Files Included

| File | Purpose |
|------|---------|
| `api/app.py` | FastAPI server with prediction, dashboard, notebook endpoints |
| `frontend/index.html` | Control Room dashboard |
| `frontend/notebook.html` | Interactive ML Lab |
| `frontend/presentation.html` | 13-slide presentation with simulator |
| `pipeline/*.py` | 8-stage ML pipeline |
| `run_pipeline.py` | Pipeline execution script |
| `requirements.txt` | Python dependencies |

---

## License

This project is for educational and demonstration purposes.

---

## Acknowledgments

- AI4I 2020 Predictive Maintenance Dataset
- Machine Predictive Maintenance Classification Dataset (Kaggle)