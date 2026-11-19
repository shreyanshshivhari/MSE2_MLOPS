"""
src/app.py
==========
PURPOSE (MLOps context):
  This is the MODEL SERVING stage — it exposes the trained model as a
  REST API so any client (browser, mobile app, another service) can
  classify documents without knowing anything about the ML internals.

  It also implements lightweight MONITORING by logging low-confidence
  predictions ("drift events") to a CSV file.  In production you would
  feed this into a dashboard (Grafana, Evidently AI, etc.).

HOW TO RUN (locally):
  uvicorn src.app:app --reload --port 8000

ENDPOINTS:
  GET  /health        → API status + known classes
  POST /predict       → Upload a .txt or .pdf, get predicted class + confidence
  GET  /monitoring    → Recent drift log entries
"""

import csv
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.extract import extract_text   # our text-extraction helper

# ── Paths ───────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "doc_classifier.joblib"
FEEDBACK_DIR = ROOT / "data" / "feedback"
DRIFT_LOG = FEEDBACK_DIR / "drift_log.csv"

# ── Confidence threshold for drift detection ────────────────────────────────────
# If the model is less than 65% confident, we flag the prediction as uncertain.
# This signals potential distribution shift (a document type the model hasn't seen).
DRIFT_THRESHOLD = 0.65

# ── Application state (holds the model after startup) ───────────────────────────
# Using a plain dict is the simplest pattern for shared mutable state in FastAPI.
_state: dict = {"pipeline": None, "classes": []}


# ── Lifespan context manager ─────────────────────────────────────────────────────
# The modern FastAPI pattern (replaces deprecated @app.on_event("startup")).
# Code before `yield` runs at startup; code after `yield` runs at shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──────────────────────────────────────────────────────────────────
    if MODEL_PATH.exists():
        artefact = joblib.load(MODEL_PATH)   # loads the dict we saved in train.py
        _state["pipeline"] = artefact["pipeline"]
        _state["classes"] = artefact["classes"]

        # Ensure the drift log directory exists
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

        # Create drift_log.csv with headers if it doesn't exist yet
        if not DRIFT_LOG.exists():
            with open(DRIFT_LOG, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "filename", "predicted_class",
                                 "confidence", "drift_alert"])

        print(f"[OK] Model loaded. Classes: {_state['classes']}")
    else:
        print(
            f"[WARNING] Model not found at {MODEL_PATH}. "
            "Run  python src/train.py  first."
        )

    yield   # ← server is running here

    # ── SHUTDOWN (cleanup if needed) ─────────────────────────────────────────────
    _state["pipeline"] = None


# ── Create the FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Document Classifier",
    description="Classify invoices, resumes, and certificates using ML.",
    version="1.0.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 1: Health check
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", summary="Check API status")
def health() -> dict:
    """
    Returns the API status and the classes the model knows about.
    Health checks are essential in MLOps – monitoring tools and
    CI/CD pipelines call this to confirm the service is live.
    """
    return {
        "status": "ok",
        "model_loaded": _state["pipeline"] is not None,
        "known_classes": _state["classes"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 2: Document prediction
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/predict", summary="Classify an uploaded document")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Accept a .txt or .pdf file, extract its text, and return:
      - predicted_class  : the most likely document type
      - confidence       : probability of the top prediction (0 – 1)
      - drift_alert      : True if confidence < 0.65 (model is uncertain)
      - all_scores       : probability for every known class
    """
    # ── Validate file type FIRST (before checking model) ─────────────────────────
    # This returns a clean 400 error regardless of whether the model is loaded.
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".pdf"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Only .txt and .pdf accepted.",
        )

    # ── Check model is loaded ─────────────────────────────────────────────────────
    if _state["pipeline"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run  python src/train.py  first.",
        )

    # ── Save uploaded file to a temp location so extract_text can read it ───────
    # Write to /tmp (automatically cleaned by OS, no disk clutter)
    tmp_path = Path(f"/tmp/upload_{file.filename}")
    content = await file.read()
    tmp_path.write_bytes(content)

    # ── Extract text ─────────────────────────────────────────────────────────────
    try:
        text = extract_text(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Text extraction failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)   # always clean up the temp file

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "Extracted text is empty. If this is a scanned photo or image PDF, "
                "please upload a PDF containing selectable text or a .txt file."
            ),
        )

    # ── Predict ───────────────────────────────────────────────────────────────────
    # predict_proba returns a 2-D array: [[p_class0, p_class1, ...]]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probas = _state["pipeline"].predict_proba([text])[0]

    top_idx = int(probas.argmax())
    predicted_class = _state["classes"][top_idx]
    confidence = float(probas[top_idx])

    # ── Drift detection ───────────────────────────────────────────────────────────
    # A low confidence score means the document doesn't look like anything
    # the model was trained on — a sign that the data distribution has changed.
    drift_alert = confidence < DRIFT_THRESHOLD

    if drift_alert:
        _log_drift(file.filename, predicted_class, confidence)

    # Build a human-readable scores dict (class_name → probability)
    all_scores = {cls: round(float(p), 4) for cls, p in zip(_state["classes"], probas)}

    return JSONResponse({
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "drift_alert": drift_alert,
        "all_scores": all_scores,
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 3: Monitoring dashboard
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/monitoring", summary="View recent drift events")
def monitoring(last_n: int = 20) -> dict:
    """
    Returns the most recent `last_n` entries from drift_log.csv and
    a summary count of how many drift alerts have been triggered.

    In a production MLOps system this data would feed a Grafana dashboard
    or trigger an automated retraining job.
    """
    if not DRIFT_LOG.exists():
        return {
            "total_predictions_logged": 0,
            "drift_alert_count": 0,
            "recent_entries": [],
        }

    rows = []
    with open(DRIFT_LOG, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    drift_count = sum(1 for r in rows if r.get("drift_alert") == "True")

    return {
        "total_predictions_logged": len(rows),
        "drift_alert_count": drift_count,
        "recent_entries": rows[-last_n:],   # most recent n entries
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Internal helper: drift logger
# ═══════════════════════════════════════════════════════════════════════════════

def _log_drift(filename: str, predicted_class: str, confidence: float) -> None:
    """
    Append a row to drift_log.csv every time a low-confidence prediction occurs.
    This creates an auditable record of model uncertainty over time.
    """
    # Ensure parent directory exists (in case feedback/ was deleted)
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),   # ISO-8601 timestamp
            filename,
            predicted_class,
            round(confidence, 4),
            True,                            # drift_alert flag
        ])
