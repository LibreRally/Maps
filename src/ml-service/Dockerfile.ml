# LibreRally.Maps ML Service — Python 3.12 + Open3D 0.19 + Open3D-ML
#
# Build:  docker build -t librerallymaps-ml -f Dockerfile.ml .
# Run:    docker run -p 8000:8000 -v ./data:/data librerallymaps-ml
#
# Open3D 0.19.0 supports Python 3.10-3.12 (not 3.13 yet).
# Open3D-ML provides RandLANet/KPFCNN for semantic point cloud segmentation.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System dependencies ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Use Python 3.12 explicitly and create virtual environment
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && python --version \
    && python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# ── Python dependencies ──────────────────────────────────────────
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch 2.2 (CPU) — matches Open3D 0.19.0
RUN python -m pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# Open3D 0.19.0 + deps
RUN python -m pip install --no-cache-dir \
    open3d==0.19.0 \
    scipy scikit-learn laspy psycopg2-binary pyproj \
    fastapi uvicorn requests cloth-simulation-filter \
    tensorboard \
    "numpy<2"

# Install Open3D-ML (copy ml3d to site-packages — avoids setup.py torch import)
RUN wget -q https://github.com/isl-org/Open3D-ML/archive/refs/heads/main.zip -O /tmp/open3dml.zip \
    && python -m zipfile -e /tmp/open3dml.zip /opt/ \
    && mv /opt/Open3D-ML-main /opt/Open3D-ML \
    && cp -r /opt/Open3D-ML/ml3d $(python -c "import site; print(site.getsitepackages()[0])")/ \
    && rm /tmp/open3dml.zip

# ── Pre-download RandLANet pretrained weights ────────────────────
RUN mkdir -p /opt/models && \
    wget -q https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth \
         -O /opt/models/randlanet_semantickitti.pth

# ── Application ──────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt 2>/dev/null; exit 0

COPY main.py classifier.py reconstruction.py labeler.py open3d_classifier.py ./

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
