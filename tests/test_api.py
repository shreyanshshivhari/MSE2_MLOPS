"""
tests/test_api.py
=================
PURPOSE (MLOps context):
  Automated tests are crucial in MLOps pipelines.
  These tests run in GitHub Actions on every push to catch regressions
  BEFORE the new code is deployed or a new Docker image is pushed.

  We use FastAPI's built-in TestClient (backed by HTTPX) so we can
  call API endpoints directly in Python without starting a real server.

HOW TO RUN:
  pytest                        # run all tests
  pytest -v                     # verbose output
  pytest tests/test_api.py -v   # just this file
"""

import io
import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app instance from src/app.py
from src.app import app

# ── Create a TestClient using context manager ───────────────────────────────────
# Using `with TestClient(app) as client:` ensures the FastAPI lifespan
# (startup/shutdown) events fire correctly — the model loads on startup
# so all predict tests get a live, ready-to-use pipeline.
# We use a module-level fixture so the client (and model) is shared
# across all tests in this file, making the test suite fast.

@pytest.fixture(scope="module")
def client():
    """
    Module-scoped fixture: start the TestClient once, yield it to all tests,
    then shut it down at the end.  'scope=module' means startup runs once per
    file, not once per test — much faster.
    """
    with TestClient(app) as c:
        yield c


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 1: Health endpoint
# ═══════════════════════════════════════════════════════════════════════════════

def test_health_returns_200(client):
    """The /health endpoint must always return HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"


def test_health_has_required_fields(client):
    """The /health response must contain 'status' and 'known_classes'."""
    response = client.get("/health")
    body = response.json()
    assert "status" in body, "Missing 'status' field in /health response"
    assert "known_classes" in body, "Missing 'known_classes' field in /health response"
    assert body["status"] == "ok", f"Expected status='ok', got '{body['status']}'"


def test_health_model_loaded(client):
    """The model should be loaded after startup (requires trained model)."""
    response = client.get("/health")
    body = response.json()
    # If model_loaded is False the pipeline hasn't been trained yet
    assert body.get("model_loaded") is True, (
        "model_loaded is False. Run  python src/train.py  before running tests."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2: Predict endpoint – happy path (text file)
# ═══════════════════════════════════════════════════════════════════════════════

def test_predict_with_txt_file(client):
    """
    POST a small .txt file and check that the response has all expected fields.
    We don't assert the specific predicted class because that depends on the
    trained model's weights — asserting the STRUCTURE is more robust.
    """
    sample_text = (
        "INVOICE #INV-2024-001  "
        "Date: 01/06/2024  "
        "Bill To: Acme Corp  "
        "Item: Software License  Qty: 1  Price: 5000.00  "
        "Total: 5900.00 (incl 18% GST)  "
        "Payment due in 30 days."
    )
    # Simulate a file upload: wrap the text as bytes in a file-like object
    file_bytes = io.BytesIO(sample_text.encode("utf-8"))

    response = client.post(
        "/predict",
        files={"file": ("test_invoice.txt", file_bytes, "text/plain")},
    )

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}. Body: {response.text}"
    )
    body = response.json()

    # Check all required fields are present
    for field in ("predicted_class", "confidence", "drift_alert", "all_scores"):
        assert field in body, f"Missing field '{field}' in /predict response"

    # Confidence must be a float between 0 and 1
    assert 0.0 <= body["confidence"] <= 1.0, (
        f"Confidence out of range: {body['confidence']}"
    )

    # drift_alert must be a boolean
    assert isinstance(body["drift_alert"], bool), (
        f"Expected drift_alert to be bool, got {type(body['drift_alert'])}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 3: Predict endpoint – unsupported file type
# ═══════════════════════════════════════════════════════════════════════════════

def test_predict_rejects_unsupported_file_type(client):
    """
    Uploading a .csv or .doc file should return HTTP 400 (Bad Request),
    not a 500 server crash.
    """
    file_bytes = io.BytesIO(b"col1,col2\nval1,val2")
    response = client.post(
        "/predict",
        files={"file": ("data.csv", file_bytes, "text/csv")},
    )
    assert response.status_code == 400, (
        f"Expected 400 for .csv upload, got {response.status_code}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 4: Monitoring endpoint
# ═══════════════════════════════════════════════════════════════════════════════

def test_monitoring_returns_200(client):
    """The /monitoring endpoint must return HTTP 200."""
    response = client.get("/monitoring")
    assert response.status_code == 200


def test_monitoring_has_required_fields(client):
    """The /monitoring response must contain expected summary fields."""
    response = client.get("/monitoring")
    body = response.json()
    assert "total_predictions_logged" in body
    assert "drift_alert_count" in body
    assert "recent_entries" in body
    assert isinstance(body["recent_entries"], list)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 5: Drift detection logic
# ═══════════════════════════════════════════════════════════════════════════════

def test_drift_alert_for_ambiguous_text(client):
    """
    Upload a very short, ambiguous text that provides little signal.
    We check that the response structure is correct regardless of drift outcome.
    (We can't guarantee a drift alert, but the field must always be present.)
    """
    ambiguous_text = "Hello. This is a document."
    file_bytes = io.BytesIO(ambiguous_text.encode("utf-8"))

    response = client.post(
        "/predict",
        files={"file": ("ambiguous.txt", file_bytes, "text/plain")},
    )
    # Should not crash — either 200 (predicted something) or we just check structure
    assert response.status_code == 200
    body = response.json()
    assert "drift_alert" in body
