# ── Dockerfile ──────────────────────────────────────────────────────────────────
# PURPOSE (MLOps context):
#   Docker packages the app + its dependencies into a single portable image.
#   This guarantees the model runs the same way on every machine, eliminating
#   "it works on my laptop" problems — a core MLOps principle.
#
# BUILD:
#   docker build -t smart-doc-classifier .
#
# RUN:
#   docker run -p 8000:8000 smart-doc-classifier
# ────────────────────────────────────────────────────────────────────────────────

# Use the official slim Python 3.11 image as the base.
# "slim" excludes many unnecessary system packages, keeping the image small.
FROM python:3.11-slim

# Set the working directory inside the container.
# All subsequent COPY / RUN commands resolve paths relative to this.
WORKDIR /app

# ── Install system dependency needed by some Python packages ────────────────────
# libgomp1 is required by scikit-learn's OpenMP parallelism on Linux.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ─────────────────────────────────────────────────
# Copy requirements.txt FIRST (before copying source code).
# Docker caches each layer: if requirements.txt hasn't changed, this expensive
# pip install step is skipped on the next build — much faster CI/CD.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source code ────────────────────────────────────────────────
# We only copy what the API needs at runtime: src/ and models/.
# Data files and training scripts are excluded via .dockerignore.
COPY src/ src/
COPY models/ models/

# ── Expose the API port ──────────────────────────────────────────────────────────
# This documents which port the container listens on.
# The actual port mapping is done with  docker run -p 8000:8000 ...
EXPOSE 8000

# ── Start the FastAPI server ─────────────────────────────────────────────────────
# uvicorn is the ASGI server that runs the FastAPI application.
# --host 0.0.0.0 means "accept connections from any IP" (not just localhost).
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
