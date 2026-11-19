"""
src/extract.py
==============
PURPOSE (MLOps context):
  The API accepts user uploads in .txt OR .pdf format.
  This helper centralises all text-extraction logic so app.py stays clean.
  Keeping I/O utilities separate from business logic is a good software practice.
"""

from pathlib import Path
import pypdf  # PDF parsing library


def extract_text(file_path: str | Path) -> str:
    """
    Extract plain text from a .txt or .pdf file.

    Parameters
    ----------
    file_path : str or Path
        Absolute or relative path to the uploaded file.

    Returns
    -------
    str
        The extracted text, stripped of leading/trailing whitespace.

    Raises
    ------
    ValueError
        If the file extension is not supported (.txt or .pdf only).
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        # Simple UTF-8 read; 'replace' avoids crashes on odd characters
        return path.read_text(encoding="utf-8", errors="replace").strip()

    elif suffix == ".pdf":
        try:
            reader = pypdf.PdfReader(str(path))
            pages_text = []
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)
                except Exception:
                    continue
            return "\n".join(pages_text).strip()
        except Exception as err:
            raise ValueError(f"Could not parse PDF file: {err}")

    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Only .txt and .pdf are accepted."
        )
