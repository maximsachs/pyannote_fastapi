# syntax=docker/dockerfile:1

ARG PYTORCH_BASE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
ARG PYANNOTE_VERSION=3.3.1

FROM ${PYTORCH_BASE}

ARG PYANNOTE_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/opt/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/huggingface

WORKDIR /app

COPY app/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt "pyannote.audio==${PYANNOTE_VERSION}"

COPY app/ /app/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/live || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
