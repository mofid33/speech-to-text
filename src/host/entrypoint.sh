#!/bin/sh
set -eu

load_runtime_environment() {
    secret_file="${VAULT_ENV_FILE:-/run/secrets/speech-to-text.env}"

    if [ -n "${VAULT_ENV_FILE:-}" ] && [ ! -r "$secret_file" ]; then
        echo "Vault runtime environment file '$secret_file' is not readable." >&2
        exit 1
    fi

    if [ -r "$secret_file" ]; then
        set -a
        # shellcheck disable=SC1090
        . "$secret_file"
        set +a
    fi
}

load_runtime_environment

export ENV_TYPE="${ENV_TYPE:-production}"

role="${1:-api}"

if [ "$#" -gt 0 ]; then
    shift
fi

case "$role" in
    migrate)
        python -m src.infrastructure.scripts.validate_runtime --role migrate
        exec python -m src.infrastructure.scripts.run_migrations --upgrade
        ;;

    api)
        python -m src.infrastructure.scripts.validate_runtime --role api
        exec uvicorn src.host.app:app \
            --host 0.0.0.0 \
            --port 8000 \
            "$@"
        ;;

    worker)
        python -m src.infrastructure.scripts.validate_runtime --role worker
        exec python -m src.infrastructure.tasks.worker \
            --loglevel="${CELERY_LOG_LEVEL:-INFO}" \
            --pool="${CELERY_WORKER_POOL:-solo}" \
            --concurrency="${CELERY_WORKER_CONCURRENCY:-1}" \
            "$@"
        ;;

    *)
        exec "$role" "$@"
        ;;
esac