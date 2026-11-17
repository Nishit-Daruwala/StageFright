# Builds natively for the host architecture (arm64 on Apple silicon, amd64 on Intel/AMD).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    MODELS_DIR=/models/stagefright

WORKDIR /app

COPY requirements.txt .
# The CPU wheel index keeps PyTorch small; the container cannot use a GPU anyway.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY server ./server
COPY web ./web

EXPOSE 8000
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
