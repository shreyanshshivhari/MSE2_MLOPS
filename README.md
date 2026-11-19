# 📄 Smart Document Classification & Monitoring

> An end-to-end MLOps project demonstrating the full ML lifecycle:
> data preparation → model training → experiment tracking → REST API → drift monitoring → Docker → CI/CD.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     MLOps PIPELINE                              │
│                                                                 │
│  1. DATA PREP          2. TRAINING         3. SERVING           │
│  ┌────────────┐        ┌──────────┐        ┌──────────────┐    │
│  │prepare_    │──raw   │          │─model─▶│  FastAPI     │    │
│  │data.py     │  .txt  │ train.py │        │  app.py      │    │
│  │            │──────▶│          │        │  /health     │    │
│  │• Kaggle CSV│        │• TF-IDF  │        │  /predict    │    │
│  │• Synthetic │        │• SVC     │        │  /monitoring │    │
│  │  fallback  │        │• MLflow  │        └──────┬───────┘    │
│  └────────────┘        └──────────┘               │            │
│                                                   │drift       │
│  4. VERSIONING         5. MONITORING         ┌────▼──────┐     │
│  ┌────────────┐        ┌──────────────┐      │drift_log  │     │
│  │  DVC       │        │  drift_      │◀─────│.csv       │     │
│  │  data/raw/ │        │  log.csv     │      └───────────┘     │
│  └────────────┘        └──────────────┘                        │
│                                                                 │
│  6. CONTAINERIZATION   7. CI/CD                                 │
│  ┌────────────┐        ┌──────────────────────────────────┐    │
│  │  Docker    │        │  GitHub Actions                   │    │
│  │  image     │        │  push/PR → pytest → Docker push  │    │
│  └────────────┘        └──────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
MSE2/
├── src/
│   ├── __init__.py          # makes src/ a Python package
│   ├── prepare_data.py      # Stage 1: generate/load data
│   ├── extract.py           # helper: PDF/TXT text extraction
│   ├── train.py             # Stage 2: train model + MLflow logging
│   └── app.py               # Stage 3: FastAPI serving + monitoring
│
├── tests/
│   └── test_api.py          # pytest tests for the API
│
├── data/
│   ├── raw/                 # clean .txt files, one folder per class
│   │   ├── invoices/
│   │   ├── resumes/
│   │   ├── certificates/
│   │   └── bank_statements/ # added in v2 retraining demo
│   ├── external/            # ← DROP YOUR KAGGLE CSVs HERE
│   └── feedback/
│       └── drift_log.csv    # auto-created when drift events occur
│
├── models/
│   └── doc_classifier.joblib  # saved pipeline + class names
│
├── sample_uploads/
│   ├── test_invoice.txt
│   └── test_unknown_bank_statement.txt
│
├── mlruns/                  # MLflow experiment logs (auto-created)
├── .github/workflows/
│   └── ci-cd.yml            # GitHub Actions CI/CD pipeline
├── Dockerfile
├── .dockerignore
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.11+
- Git
- Docker (optional, for containerisation demo)

### 1 — Clone and set up environment

```bash
git clone <your-repo-url>
cd MSE2
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — (Optional) Add Kaggle datasets

Download from Kaggle and place CSVs in `data/external/`:

| Class | Kaggle Dataset | File to place in `data/external/` |
|-------|---------------|----------------------------------|
| Invoices | `nikitpatel/invoice-ner-dataset` | `invoice_ner.csv` (has an `Input` column) |
| Resumes | `snehaanbhawal/resume-dataset` | `resume_dataset.csv` (has a `Resume_str` column) |

> **If you skip this step**, the script automatically generates realistic synthetic samples — the pipeline runs either way.

### 3 — Prepare data

```bash
# Basic: 3 classes (invoices, resumes, certificates) with 120 docs each
python src/prepare_data.py

# With bank statements (v2 drift demo): 4 classes
python src/prepare_data.py --add-bank-statements

# Smaller dataset (faster CI):
python src/prepare_data.py --n 30
```

### 4 — Initialize Git + DVC (data versioning)

```bash
git init
dvc init
dvc add data/raw          # DVC tracks the data folder
git add .gitignore data/raw.dvc .dvc/
git commit -m "feat: initial data version v1 (3 classes)"
```

### 5 — Train the model

```bash
python src/train.py
```

The script prints accuracy and F1-score, logs everything to MLflow, and saves the model to `models/doc_classifier.joblib`.

### 6 — View MLflow experiment UI

```bash
mlflow ui --port 5000
# Open: http://localhost:5000
```

You'll see the run with all logged parameters, metrics, and the saved model artefact.

### 7 — Run the API locally

```bash
uvicorn src.app:app --reload --port 8000
```

**Test the endpoints:**

```bash
# Health check
curl http://localhost:8000/health

# Predict (upload a file)
curl -X POST http://localhost:8000/predict \
  -F "file=@sample_uploads/test_invoice.txt"

# Predict with the unknown bank statement (triggers drift alert)
curl -X POST http://localhost:8000/predict \
  -F "file=@sample_uploads/test_unknown_bank_statement.txt"

# View monitoring/drift log
curl http://localhost:8000/monitoring
```

### 8 — Run automated tests

```bash
pytest                    # all tests
pytest -v                 # verbose output
```

---

## 🐳 Docker

```bash
# Build the image
docker build -t smart-doc-classifier .

# Run the container
docker run -p 8000:8000 smart-doc-classifier

# Test it
curl http://localhost:8000/health
```

> **Note:** The Docker image includes `src/` and `models/` only.
> Training data is excluded via `.dockerignore` to keep the image small.

---

## 🔄 Retraining Demo (v2 — Adding a New Class)

This demonstrates a core MLOps concept: the model must be **retrained and redeployed** when new data types appear.

```bash
# Step 1: Add bank_statements class
python src/prepare_data.py --add-bank-statements

# Step 2: Version the new data with DVC
dvc add data/raw
git add data/raw.dvc
git commit -m "feat: add bank_statements class (v2 data)"

# Step 3: Retrain → MLflow logs a NEW run you can compare with v1
python src/train.py

# Step 4: See both runs in MLflow UI
mlflow ui --port 5000
```

---

## ⚙️ GitHub Actions CI/CD

The pipeline in `.github/workflows/ci-cd.yml` runs automatically on every push to `main`:

| Job | Trigger | What it does |
|-----|---------|-------------|
| `test` | every push + PR | installs deps → generates 30 synthetic docs → trains → runs `pytest` |
| `build-and-push` | push to `main` only (after tests pass) | logs into Docker Hub → builds image → pushes with `:latest` and `:<commit-sha>` tags |

**Required GitHub Secrets** (Settings → Secrets → Actions):
- `DOCKERHUB_USERNAME` — your Docker Hub username
- `DOCKERHUB_TOKEN` — Docker Hub access token (create at hub.docker.com → Account Settings → Security)

---

## 🎓 Teacher Demo Script (Step-by-Step)

| # | What to show | Command / action |
|---|-------------|-----------------|
| 1 | **Project structure** | `tree . -L 3 --gitignore` |
| 2 | **Data preparation** | `python src/prepare_data.py` → show files in `data/raw/` |
| 3 | **Data versioning** | `dvc init && dvc add data/raw && git add . && git commit -m "v1 data"` |
| 4 | **Training + MLflow** | `python src/train.py` → `mlflow ui --port 5000` → open browser |
| 5 | **Start API** | `uvicorn src.app:app --port 8000` |
| 6 | **Predict invoice** | `curl -X POST localhost:8000/predict -F "file=@sample_uploads/test_invoice.txt"` |
| 7 | **Trigger drift alert** | `curl -X POST localhost:8000/predict -F "file=@sample_uploads/test_unknown_bank_statement.txt"` |
| 8 | **View drift log** | `curl localhost:8000/monitoring` or `cat data/feedback/drift_log.csv` |
| 9 | **Run tests** | `pytest -v` |
| 10 | **Docker** | `docker build -t smart-doc-classifier . && docker run -p 8000:8000 smart-doc-classifier` |
| 11 | **Retraining v2** | `python src/prepare_data.py --add-bank-statements && python src/train.py` → compare MLflow runs |
| 12 | **CI/CD** | Show `.github/workflows/ci-cd.yml` and GitHub Actions run |

---

## 🔑 Key MLOps Concepts Covered

| Concept | Implementation in this project |
|---------|-------------------------------|
| **Data versioning** | DVC tracks `data/raw/` — every training run is reproducible |
| **Experiment tracking** | MLflow logs params, metrics, and model artefacts per run |
| **Model serving** | FastAPI exposes a REST API so any client can query the model |
| **Drift monitoring** | Confidence < 0.65 → logged to `drift_log.csv` → visible at `/monitoring` |
| **Containerisation** | Docker packages the app + model into a portable, reproducible image |
| **CI/CD** | GitHub Actions tests every PR and auto-pushes Docker images to Docker Hub |
| **Retraining workflow** | Adding `bank_statements` class and re-running train.py = new MLflow run |

---

## 📋 Requirements

See [requirements.txt](requirements.txt) for exact versions.

| Package | Purpose |
|---------|---------|
| `scikit-learn` | TF-IDF vectoriser + SVC classifier |
| `joblib` | Save/load the trained model pipeline |
| `mlflow` | Experiment tracking and model registry |
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server to run FastAPI |
| `python-multipart` | File uploads in FastAPI |
| `pypdf` | Extract text from PDF uploads |
| `pytest` + `httpx` | Automated API testing |
| `dvc` | Data Version Control |
