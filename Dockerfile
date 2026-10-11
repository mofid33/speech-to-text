# syntax=docker/dockerfile:1.7

FROM python:3.11-slim-bookworm AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/appuser \
    HF_HOME=/var/lib/speech-to-text/huggingface \
    XDG_CACHE_HOME=/var/lib/speech-to-text/cache

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements.txt

RUN groupadd --gid 10001 appgroup \
    && useradd \
        --uid 10001 \
        --gid 10001 \
        --create-home \
        --shell /usr/sbin/nologin \
        appuser \
    && mkdir -p \
        /var/lib/speech-to-text/models \
        /var/lib/speech-to-text/huggingface \
        /var/lib/speech-to-text/cache \
    && chown -R appuser:appgroup \
        /var/lib/speech-to-text \
        /home/appuser


FROM runtime-base AS api

COPY --chown=appuser:appgroup src /app/src

RUN chmod 0755 /app/src/host/entrypoint.sh

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health_check/ready', timeout=3).read()"

STOPSIGNAL SIGTERM

ENTRYPOINT ["/app/src/host/entrypoint.sh"]
CMD ["api"]


FROM runtime-base AS worker

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.worker.txt /app/requirements.worker.txt

RUN python -m pip install -r /app/requirements.worker.txt

COPY --chown=appuser:appgroup src /app/src

RUN chmod 0755 /app/src/host/entrypoint.sh

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

USER appuser

EXPOSE 9101

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9101/metrics', timeout=3).read()"

STOPSIGNAL SIGTERM

ENTRYPOINT ["/app/src/host/entrypoint.sh"]
CMD ["worker"]