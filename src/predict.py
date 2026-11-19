"""
src/predict.py
==============
PURPOSE:
  Quick CLI tool to test any custom document (.pdf or .txt) directly
  without needing to launch the FastAPI server.

HOW TO RUN:
  python src/predict.py path/to/your/document.pdf
"""

import sys
from pathlib import Path

# Add project root to sys.path so "from src.extract" works from anywhere
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
from src.extract import extract_text
MODEL_PATH = ROOT / "models" / "doc_classifier.joblib"


def predict_document(file_path: str | Path) -> None:
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    if not MODEL_PATH.exists():
        print(f"❌ Model not found at {MODEL_PATH}. Please run: python src/train.py")
        return

    # 1. Load model and class names
    artefact = joblib.load(MODEL_PATH)
    pipeline = artefact["pipeline"]
    classes = artefact["classes"]

    # 2. Extract text from PDF or TXT
    print(f"\n📄 Inspecting: {path.name}")
    try:
        text = extract_text(path)
    except Exception as e:
        print(f"❌ Failed to extract text: {e}")
        return

    if not text.strip():
        print("⚠️ Warning: Extracted text is empty! (If this is a scanned PDF image, OCR is required).")
        return

    print(f"📝 Extracted {len(text)} characters.")
    print("─── Preview of Extracted Text ─────────────────────────────")
    print(text[:300] + ("..." if len(text) > 300 else ""))
    print("──────────────────────────────────────────────────────────")

    # 3. Model prediction & probabilities
    probas = pipeline.predict_proba([text])[0]
    top_idx = int(probas.argmax())
    predicted_class = classes[top_idx]
    confidence = float(probas[top_idx])

    # 4. Display results
    print("\n🎯 PREDICTION RESULTS:")
    print(f"  • Predicted Class : {predicted_class.upper()}")
    print(f"  • Confidence      : {confidence * 100:.2f}%")
    if confidence < 0.65:
        print("  • Drift Alert     : ⚠️ YES (Confidence < 65% — Out of Distribution document)")
    else:
        print("  • Drift Alert     : ✅ NO (High confidence match)")

    print("\n📊 Probability Breakdown:")
    for cls_name, prob in sorted(zip(classes, probas), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 30)
        print(f"  {cls_name:<16} : {prob * 100:>6.2f}% | {bar}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_file.pdf_or_txt>")
        sys.exit(1)

    predict_document(sys.argv[1])
