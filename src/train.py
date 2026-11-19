"""
src/train.py
============
PURPOSE (MLOps context):
  This is the MODEL TRAINING stage of the pipeline.
  It reads the prepared .txt files, trains a scikit-learn Pipeline,
  logs everything to MLflow (experiment tracking), and saves the model
  as a joblib file (model registry / artefact store).

HOW TO RUN:
  python src/train.py

  After training, view the MLflow UI with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
  Then open http://localhost:5000 in your browser.
"""

import os
from pathlib import Path

import joblib                               # save/load Python objects to disk
import mlflow                               # experiment tracking
import mlflow.sklearn                       # sklearn-specific MLflow helpers
from sklearn.datasets import load_files     # loads a folder-of-folders as a dataset
from sklearn.feature_extraction.text import TfidfVectorizer  # text → numbers
from sklearn.metrics import accuracy_score, f1_score          # evaluation
from sklearn.model_selection import train_test_split          # split data
from sklearn.pipeline import Pipeline                         # chain steps
from sklearn.svm import SVC                                   # classifier

# ── Paths ───────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "raw"           # one sub-folder per class
MODEL_PATH = ROOT / "models" / "doc_classifier.joblib"

# ── Hyperparameters (easy to change and compare via MLflow) ─────────────────────
MAX_FEATURES = 10_000   # TF-IDF vocabulary size; more = slower but potentially better
SVC_KERNEL = "linear"   # linear kernel works well for text classification
TEST_SIZE = 0.2         # 20% of data held out for evaluation
RANDOM_STATE = 42       # fixed seed → reproducible train/test split

# ── MLflow trusted types for skops serialisation (required in MLflow 3.x) ───────
# MLflow 3.x uses skops to safely serialise sklearn models.
# We explicitly list the internal sklearn/scipy types our pipeline uses.
SKOPS_TRUSTED = [
    "sklearn.pipeline.Pipeline",
    "sklearn.feature_extraction.text.TfidfVectorizer",
    "sklearn.svm._classes.SVC",
    "sklearn.calibration.CalibratedClassifierCV",
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration",
    "scipy.sparse._csr.csr_matrix",
    "numpy.ndarray",
    "numpy.float64",
    "numpy.int64",
    "builtins.NoneType",
]


def train() -> None:
    """
    Full training workflow:
      1. Load data  →  2. Split  →  3. Build pipeline  →  4. Train
      →  5. Evaluate  →  6. Log to MLflow  →  7. Save model
    """

    # ── STEP 1: Load raw .txt files ─────────────────────────────────────────────
    # load_files treats every SUB-FOLDER of DATA_DIR as a class label.
    # It returns a Bunch with .data (list of bytes), .target (int labels),
    # and .target_names (list of class name strings).
    print(f"[1/5] Loading documents from: {DATA_DIR}")
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        raise FileNotFoundError(
            f"No data found in {DATA_DIR}. "
            "Please run:  python src/prepare_data.py"
        )

    dataset = load_files(
        str(DATA_DIR),
        encoding="utf-8",
        decode_error="replace",
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    class_names: list[str] = dataset.target_names
    num_docs = len(dataset.data)
    print(f"   Classes  : {class_names}")
    print(f"   Total docs: {num_docs}")

    # ── STEP 2: Train / test split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,       # list of raw text strings
        dataset.target,     # list of integer class indices
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataset.target,   # preserve class balance in both splits
    )
    print(f"[2/5] Train size: {len(X_train)} | Test size: {len(X_test)}")

    # ── STEP 3: Build sklearn Pipeline ──────────────────────────────────────────
    # A Pipeline chains transformers + an estimator so the same object
    # handles both fit() and predict() — no risk of data leakage.
    pipeline = Pipeline([
        # TF-IDF converts raw text to a sparse numerical feature matrix.
        # It down-weights words common across ALL documents (like "the")
        # and up-weights rare but discriminative words.
        ("tfidf", TfidfVectorizer(
            max_features=MAX_FEATURES,
            sublinear_tf=True,     # use log(1+tf) instead of tf → helps with long docs
            strip_accents="unicode",
            ngram_range=(1, 2),    # unigrams + bigrams capture short phrases
        )),
        # Linear SVC is fast, interpretable, and excellent for text classification.
        # probability=True enables predict_proba() for confidence scores.
        # Note: in scikit-learn 1.9+ this triggers a FutureWarning; for a
        # beginner-friendly project we keep it here and suppress the warning.
        ("svc", SVC(
            kernel=SVC_KERNEL,
            probability=True,
            random_state=RANDOM_STATE,
        )),
    ])

    # ── STEP 4: Train ────────────────────────────────────────────────────────────
    print("[3/5] Training pipeline ...")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress scikit-learn FutureWarning
        pipeline.fit(X_train, y_train)

    # ── STEP 5: Evaluate ─────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"[4/5] Accuracy : {acc:.4f}")
    print(f"      F1-Score : {f1:.4f}")

    # ── STEP 6: Log to MLflow ────────────────────────────────────────────────────
    # MLflow stores every run's parameters, metrics, and artefacts.
    # This lets us compare v1 (3 classes) vs v2 (4 classes) later.
    # We use SQLite as the tracking backend (required by MLflow 3.x;
    # the old file-store backend is deprecated).
    db_path = ROOT / "mlflow.db"
    artifacts_dir = ROOT / "mlruns"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    # Set or create experiment with local relative artifact location
    exp = mlflow.get_experiment_by_name("smart-document-classifier")
    if exp is None:
        exp_id = mlflow.create_experiment(
            "smart-document-classifier",
            artifact_location=artifacts_dir.as_uri(),
        )
    else:
        exp_id = exp.experiment_id

    with mlflow.start_run(experiment_id=exp_id):
        # Log hyperparameters
        mlflow.log_param("max_features", MAX_FEATURES)
        mlflow.log_param("kernel", SVC_KERNEL)
        mlflow.log_param("num_classes", len(class_names))
        mlflow.log_param("dataset_size", num_docs)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("ngram_range", "(1,2)")

        # Log evaluation metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log the trained sklearn pipeline as an MLflow artefact.
        # skops_trusted_types tells MLflow which internal types are safe to load.
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=SKOPS_TRUSTED,
        )

        # Also log the class names list as a plain-text artefact
        # so we always know which labels this run's model supports
        mlflow.log_text(str(class_names), "class_names.txt")

        run_id = mlflow.active_run().info.run_id
        print(f"[MLflow] Run ID : {run_id}")

    # ── STEP 7: Save model locally (for Docker / API) ────────────────────────────
    # The FastAPI app loads this file on startup.
    # We save BOTH the pipeline and the class name list together as one dict.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "classes": class_names}, MODEL_PATH)
    print(f"[5/5] Model saved → {MODEL_PATH.relative_to(ROOT)}")

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  View results:  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
    print(f"  Then visit:    http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    train()
