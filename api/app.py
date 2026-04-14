from __future__ import annotations

import base64
import html
import json
import os
from pathlib import Path
import sys
import threading
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO

import joblib
import matplotlib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "03_training_dataset" / "test_data.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "05_best_model.joblib"
PAYLOAD_PATH = FRONTEND_DIR / "assets" / "data" / "dashboard_payload.json"
SUMMARY_PATH = FRONTEND_DIR / "model_summary.json"
RAW_NOTEBOOK_PATH = PROJECT_ROOT / "Maintenance_Complete_Pipeline.ipynb"

NOTEBOOK_LOCK = threading.Lock()
NOTEBOOK_GLOBALS: dict[str, object] = {}


app = FastAPI(title="Predictive Maintenance Intelligence API", version="1.0.0")
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")
app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")


class PredictionRequest(BaseModel):
    air_temp_k: float
    process_temp_k: float
    rotational_speed: float
    torque: float
    tool_wear: float
    product_type: str = "M"
    source_dataset: str = "merged"


def _init_notebook_globals() -> dict[str, object]:
    import numpy as np

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(PROJECT_ROOT / "pipeline") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

    from pipeline.workflow import canonicalize_frames, clean_dataset, engineer_features, feature_columns, load_raw_frames

    return {
        "__name__": "__notebook__",
        "PROJECT_ROOT": PROJECT_ROOT,
        "Path": Path,
        "json": json,
        "pd": pd,
        "np": np,
        "load_raw_frames": load_raw_frames,
        "canonicalize_frames": canonicalize_frames,
        "clean_dataset": clean_dataset,
        "engineer_features": engineer_features,
        "feature_columns": feature_columns,
    }


def _get_notebook_globals(reset: bool = False) -> dict[str, object]:
    global NOTEBOOK_GLOBALS
    if reset or not NOTEBOOK_GLOBALS:
        NOTEBOOK_GLOBALS = _init_notebook_globals()
    return NOTEBOOK_GLOBALS


def _resolve_source_path(relative_path: str) -> Path:
    candidate = (PROJECT_ROOT / relative_path).resolve()
    if PROJECT_ROOT not in candidate.parents and candidate != PROJECT_ROOT:
        raise HTTPException(status_code=400, detail="Path escapes project root")
    if candidate.suffix.lower() not in {".py", ".html", ".js", ".css", ".json", ".md", ".ipynb"}:
        raise HTTPException(status_code=400, detail="Unsupported source file type")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Source file not found")
    return candidate


def _render_notebook_html(file_path: Path) -> str:
    notebook = json.loads(file_path.read_text(encoding="utf-8"))
    cells_html: list[str] = []
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        source = html.escape("".join(cell.get("source", [])))
        cell_type = cell.get("cell_type", "code")
        if cell_type == "markdown":
            cells_html.append(f'<section class="nb-cell nb-markdown"><div class="nb-body"><pre>{source}</pre></div></section>')
            continue
        outputs_html: list[str] = []
        for output in cell.get("outputs", []):
            if "text" in output:
                outputs_html.append(f'<pre class="nb-output">{html.escape("".join(output.get("text", [])))}</pre>')
        cells_html.append(f'<section class="nb-cell nb-code"><div class="nb-label">Cell {index}</div><pre class="nb-code-block"><code>{source}</code></pre>{"".join(outputs_html)}</section>')
    body = "\n".join(cells_html)
    title = file_path.stem.replace("_", " ")
    return f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title><style>:root{{color-scheme:dark}}body{{margin:0;font-family:Inter,Arial,sans-serif;background:#07090f;color:#e5e7eb}}.wrap{{max-width:1080px;margin:0 auto;padding:40px 24px 80px}}.hero{{margin-bottom:32px;padding:28px 30px;border:1px solid rgba(255,255,255,0.08);border-radius:22px;background:linear-gradient(180deg, rgba(20,24,35,0.95), rgba(10,12,18,0.95));box-shadow:0 20px 80px rgba(0,0,0,0.35)}}.eyebrow{{display:inline-block;font-size:12px;letter-spacing:0.16em;text-transform:uppercase;color:#f87171;margin-bottom:12px}}h1{{margin:0 0 10px;font-size:clamp(30px,4vw,48px)}}.sub{{margin:0;color:#a1a1aa;font-size:16px}}.nb-cell{{margin:0 0 18px;border:1px solid rgba(255,255,255,0.08);border-radius:20px;background:rgba(14,18,27,0.92);overflow:hidden}}.nb-body,.nb-code-block,.nb-output{{padding:20px 22px;white-space:pre-wrap}}.nb-label{{padding:14px 18px;font-size:12px;letter-spacing:0.12em;text-transform:uppercase;color:#fca5a5;border-bottom:1px solid rgba(255,255,255,0.06)}}.nb-code-block{{background:#0b1020}}.nb-output{{border-top:1px solid rgba(255,255,255,0.06);background:#121826}}</style></head><body><main class="wrap"><header class="hero"><div class="eyebrow">Notebook Render</div><h1>{html.escape(title)}</h1><p class="sub">Rendered notebook view for the raw project notebook file.</p></header>{body}</main></body></html>'''


def _load_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact not found. Run pipeline first.")
    return joblib.load(MODEL_PATH)


def _prepare_frame(payload: PredictionRequest, numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
    row = {
        "air_temp_k": payload.air_temp_k,
        "process_temp_k": payload.process_temp_k,
        "rotational_speed_rpm": payload.rotational_speed,
        "torque_nm": payload.torque,
        "tool_wear_min": payload.tool_wear,
        "product_type": payload.product_type,
        "source_dataset": payload.source_dataset,
    }
    row["temp_delta_k"] = row["process_temp_k"] - row["air_temp_k"]
    row["power_proxy"] = row["rotational_speed_rpm"] * row["torque_nm"]
    row["wear_power_ratio"] = row["tool_wear_min"] / (abs(row["power_proxy"]) + 1.0)
    row["torque_speed_ratio"] = row["torque_nm"] / (abs(row["rotational_speed_rpm"]) + 1.0)
    row["thermal_load"] = row["temp_delta_k"] * row["torque_nm"]
    row["wear_temp_stress"] = row["tool_wear_min"] * row["temp_delta_k"]
    row["high_torque_flag"] = int(row["torque_nm"] >= 55)
    row["low_speed_flag"] = int(row["rotational_speed_rpm"] <= 1400)
    wear = row["tool_wear_min"]
    row["wear_risk_band"] = "fresh" if wear <= 50 else "moderate" if wear <= 120 else "elevated" if wear <= 180 else "critical"
    return pd.DataFrame([{key: row[key] for key in numeric_features + categorical_features}])


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/notebook")
def notebook() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "notebook.html")


@app.get("/presentation")
def presentation() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "presentation.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model_summary")
def model_summary() -> dict:
    if not SUMMARY_PATH.exists():
        raise HTTPException(status_code=503, detail="Model summary not found. Run pipeline first.")
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@app.get("/api/source")
def source(path: str = Query(..., min_length=1)) -> dict:
    file_path = _resolve_source_path(path)
    return {"path": path, "content": file_path.read_text(encoding="utf-8")}


@app.get("/api/source/raw")
def source_raw(path: str = Query(..., min_length=1)) -> FileResponse:
    file_path = _resolve_source_path(path)
    return FileResponse(file_path)


@app.get("/api/source/rendered", response_class=HTMLResponse)
def source_rendered(path: str = Query(..., min_length=1)) -> HTMLResponse:
    file_path = _resolve_source_path(path)
    if file_path.suffix.lower() != ".ipynb":
        raise HTTPException(status_code=400, detail="Rendered view is only available for notebook files")
    return HTMLResponse(_render_notebook_html(file_path))


@app.post("/run-cell")
def run_cell(payload: dict[str, object]) -> dict:
    code = str(payload.get("code", ""))
    reset = bool(payload.get("reset", False))
    with NOTEBOOK_LOCK:
        namespace = _get_notebook_globals(reset=reset)
        if reset and not code.strip():
            return {"stdout": "Kernel reset.", "stderr": "", "plot": None}

        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        plot_b64 = None
        import matplotlib.pyplot as plt
        previous_cwd = Path.cwd()
        plt.close("all")
        try:
            os.chdir(PROJECT_ROOT)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                    exec(code, namespace, namespace)
            figure_numbers = plt.get_fignums()
            if figure_numbers:
                figure = plt.figure(figure_numbers[-1])
                image_buffer = BytesIO()
                figure.savefig(image_buffer, format="png", bbox_inches="tight")
                image_buffer.seek(0)
                plot_b64 = base64.b64encode(image_buffer.read()).decode("utf-8")
        except Exception:
            traceback.print_exc(file=stderr_buffer)
        finally:
            os.chdir(previous_cwd)
            plt.close("all")

        return {"stdout": stdout_buffer.getvalue(), "stderr": stderr_buffer.getvalue(), "plot": plot_b64}


@app.get("/dashboard")
def dashboard() -> dict:
    if not PAYLOAD_PATH.exists():
        raise HTTPException(status_code=503, detail="Dashboard payload not found. Run pipeline first.")
    return json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))


@app.get("/sample_observations")
def sample_observations(limit: int = 6) -> list[dict]:
    if not DATASET_PATH.exists():
        raise HTTPException(status_code=503, detail="Processed dataset not found. Run pipeline first.")
    df = pd.read_csv(DATASET_PATH)
    cols = [
        "product_type",
        "air_temp_k",
        "process_temp_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "machine_failure",
        "failure_family",
    ]
    available = [col for col in cols if col in df.columns]
    return df[available].head(limit).to_dict(orient="records")


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict:
    artifact = _load_artifact()
    frame = _prepare_frame(payload, artifact["numeric_features"], artifact["categorical_features"])
    probability = float(artifact["pipeline"].predict_proba(frame)[0, 1])
    prediction = int(probability >= artifact["threshold"])
    return {
        "model": artifact["model_name"],
        "threshold": artifact["threshold"],
        "failure_probability": probability,
        "predicted_failure": prediction,
        "recommended_action": "inspect immediately" if probability >= max(0.65, artifact["threshold"]) else "schedule inspection" if probability >= max(0.35, artifact["threshold"] * 0.7) else "continue monitoring",
        "risk_band": "critical" if probability >= max(0.65, artifact["threshold"]) else "warning" if probability >= max(0.35, artifact["threshold"] * 0.7) else "stable",
    }
