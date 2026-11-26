FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# OpenCV headless still needs libglib; no torch/CUDA -> small image.
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY weights/ppe_s960.onnx weights/head_cls.onnx weights/
COPY assets/demo_site.mp4 assets/
ENV MINESAFE_CAMERAS="demo=assets/demo_site.mp4"

RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "minesafe.api:app", "--host", "0.0.0.0", "--port", "8000"]
