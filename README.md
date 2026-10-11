# Speech-to-Text Service

Persian speech-to-text processing with a FastAPI API, asynchronous Celery worker, PostgreSQL, RabbitMQ, S3-compatible object storage, Faster Whisper, and optional LLM-based transcript formatting.

**Project version:** `0.1.0` (as configured in the current application settings)  
**Current handoff scope:** Local development, evaluation, and testing.  
**Production status:** Deployment artifacts exist, but production deployment and end-to-end verification on the target infrastructure have **not** been completed.

This README documents the functionality currently present in the repository and provides a reproducible local test path. It does not claim that the ASR model has been selected through a production-scale benchmark.

## 1. What is implemented

- Upload an audio file through the HTTP API.
- Store the audio in MinIO (S3-compatible object storage), not PostgreSQL.
- Store metadata, job state, transcripts, and timestamped segments in PostgreSQL.
- Queue work through RabbitMQ and process it asynchronously with Celery.
- Validate audio with `ffprobe` before ASR.
- Transcribe Persian audio using `faster-whisper` / CTranslate2.
- Preserve separate `raw_text`, `normalized_text`, and `processed_text` values.
- Apply deterministic Persian normalization independently of the LLM.
- Optionally use an OpenAI-compatible LLM for conservative punctuation and formatting.
- Expose job status, final result, retry, and delete endpoints.
- Provide Alembic migrations, automated tests, structured logging, metrics, and a local end-to-end verification script.

The code uses one codebase and two processes: **API** and **Worker**.

## 2. Architecture

```text
Client
  |
  v
FastAPI API
  |----> PostgreSQL (metadata, status, transcripts)
  |----> MinIO / S3 (audio objects)
  |
  v
RabbitMQ
  |
  v
Celery Worker
  |
  v
ffprobe validation
  |
  v
Faster Whisper (ASR)
  |
  +--> raw_text (preserved)
  |
  v
Persian normalization
  |
  +--> normalized_text
  |
  v
Optional LLM post-processing
  |
  +--> processed_text
  |
  v
PostgreSQL
```

The project's pragmatic layer convention is:

```text
Host -> Application -> Core -> Infrastructure
```

- `src/host/`: application startup and configuration.
- `src/application/`: FastAPI controllers, routes, and Pydantic DTOs.
- `src/core/services/`: transcription and text-processing workflows.
- `src/infrastructure/`: repositories, SQLAlchemy models, database context, DI, providers, worker tasks, migrations, and instrumentation.

Dependencies are supplied through constructor injection and `@inject`. Database queries belong in repositories; HTTP handling belongs in controllers. The project does not introduce separate DDD/CQRS/microservice layers.

## 3. Prerequisites for local testing

The documented local path uses **Windows PowerShell**, Python **3.11**, and CPU-based ASR. A GPU is **not required** for this path.

Install:

1. Python 3.11 and `pip`.
2. Git.
3. Docker Desktop with Docker Compose.
4. FFmpeg, including the `ffprobe` executable available on `PATH`.

Check these prerequisites from a terminal:

```powershell
python --version
docker compose version
ffprobe -version
```

**Local ASR model:** The development worker is configured to use the Persian fine-tuned model [`nezamisafa/whisper-persian-v4`](https://huggingface.co/nezamisafa/whisper-persian-v4), converted to CTranslate2 at `models/whisper-persian-v4-ct2`. It runs on CPU with `compute_type: int8`. Unlike `large-v3-turbo`, the model **will not be downloaded and converted automatically on first upload**. Complete the preparation step below on each developer machine before processing audio. Source weights are about 6.18 GB; conversion and CPU inference additionally require substantial disk space and memory.

**Local LLM default:** `llm.enabled: false`. No vLLM or external LLM server is required for the default local test. In that mode, `processed_text` is expected to equal `normalized_text`.

## 4. Local setup (Windows PowerShell)

Run all commands from the repository root (the directory containing `docker-compose.yml`). If you are using the company mirror, clone or check out that repository before starting.

### Step 1 — Python environment

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 2 — Select development configuration

In **each new PowerShell terminal** that runs Python commands for the project:

```powershell
$env:ENV_TYPE = "development"
```

The local settings are read from:

```text
src/host/res/appsettings.development.yaml
```

The checked-in development settings use local-only example credentials for Docker services. **Do not reuse those credentials for a company or production deployment.**

### Step 2.5 — Prepare Persian Whisper V4 (required before the first transcription)

The checked-in Local configuration points to the converted model directory:

```yaml
asr:
  faster_whisper:
    model: "models/whisper-persian-v4-ct2"
    device: "cpu"
    compute_type: "int8"
```

The upstream `nezamisafa/whisper-persian-v4` checkpoint uses Hugging Face Transformers weights. The existing `FasterWhisperProvider` requires **CTranslate2** weights instead. These commands prepare the upstream checkpoint **without switching the application's ASR provider or storing model weights in Git**.

Run from the repository root using the activated Python 3.11 virtual environment:

```powershell
python -m pip install -r requirements.asr-conversion.txt
New-Item -ItemType Directory -Force models | Out-Null
ct2-transformers-converter --model nezamisafa/whisper-persian-v4 --output_dir models/whisper-persian-v4-ct2 --copy_files preprocessor_config.json --quantization int8
python -c "from pathlib import Path; from transformers import AutoTokenizer; p=Path('models/whisper-persian-v4-ct2'); assert (p/'model.bin').is_file(), 'CTranslate2 model.bin not found'; tokenizer=AutoTokenizer.from_pretrained('nezamisafa/whisper-persian-v4', use_fast=True); assert tokenizer.is_fast, 'A fast tokenizer is required'; tokenizer.backend_tokenizer.save(str(p/'tokenizer.json'))"
python -c "from faster_whisper import WhisperModel; WhisperModel('models/whisper-persian-v4-ct2', device='cpu', compute_type='int8'); print('Persian Whisper V4 model loaded')"
```

The upstream model repository does not include a ready-made `tokenizer.json`; the `AutoTokenizer` command exports one matching the source checkpoint. Keep the resulting model directory in `models/` (already excluded by `.gitignore`). Do not point `FasterWhisperProvider` directly at the original Transformers repository: it is not a CTranslate2 package.

The initial download/conversion is large and can require significant RAM. If any command fails, **do not continue to the audio upload step**: review the conversion error first. The model load check confirms loading only; run a real audio transcription afterward to check operational compatibility. This model choice is for the requested Local testing, not an evidence-based final Production selection.

### Step 3 — Start PostgreSQL, RabbitMQ, and MinIO

```powershell
docker compose up -d
docker compose ps
```

Local Docker ports:

| Service | URL / Port |
| --- | --- |
| PostgreSQL | `127.0.0.1:15432` |
| RabbitMQ AMQP | `127.0.0.1:25673` |
| RabbitMQ Management UI | `http://127.0.0.1:25674` |
| MinIO S3 API | `http://127.0.0.1:19000` |
| MinIO Console | `http://127.0.0.1:19001` |

The Compose file creates the MinIO `transcriptions` bucket through its initialization service. Local credentials are defined in `docker-compose.yml`.

### Step 4 — Apply database migrations

```powershell
python -m src.infrastructure.scripts.run_migrations --upgrade
```

The existing migration creates the `transcription` and `transcription_segment` tables. Use `--upgrade` for local setup; do not generate a new migration just to run the repository.

### Step 5 — Start the API

Open a **new terminal**, activate the same virtual environment, and set `ENV_TYPE=development`:

```powershell
.\.venv\Scripts\Activate.ps1
$env:ENV_TYPE = "development"
python -m uvicorn src.host.app:app --host 127.0.0.1 --port 8300
```

Local URLs:

| Endpoint | URL |
| --- | --- |
| Swagger/OpenAPI | `http://127.0.0.1:8300/docs` |
| Liveness | `http://127.0.0.1:8300/api/v1/health_check/` |
| Readiness (checks PostgreSQL) | `http://127.0.0.1:8300/api/v1/health_check/ready` |
| API metrics | `http://127.0.0.1:8300/metrics` |

> **Port note:** The current development YAML sets `server.port: 8300`. Some older README examples used port `8000` through an explicit Uvicorn argument. This guide consistently uses **8300**, matching the checked-in development configuration.

Verify the API:

```powershell
curl.exe -i http://127.0.0.1:8300/api/v1/health_check/
curl.exe -i http://127.0.0.1:8300/api/v1/health_check/ready
```

Both endpoints should return HTTP `200` in a healthy local environment.

### Step 6 — Start the transcription Worker

Open **another terminal**, activate the same environment, and run:

```powershell
.\.venv\Scripts\Activate.ps1
$env:ENV_TYPE = "development"
python -m src.infrastructure.tasks.worker --loglevel=INFO --pool=solo
```

Keep the API and Worker terminals running during audio processing. On Windows, the `solo` pool avoids relying on Unix-style process forking.

The Worker consumes the `transcription` queue. Worker metrics are configured locally at `http://127.0.0.1:9101/`.

## 5. Test transcription end to end

The repository includes development audio under:

```text
benchmarks/asr/audio/
```

For a quick test, use `benchmarks/asr/audio/sample-001.wav`, or a speech recording that you have permission to process.

Open a **third PowerShell terminal** at the repository root.

### Upload an audio file

```powershell
$upload = curl.exe -sS -F "file=@benchmarks/asr/audio/sample-001.wav" "http://127.0.0.1:8300/api/v1/transcription/" | ConvertFrom-Json
$ID = $upload.id
$upload
```

Expected response: HTTP `202 Accepted` with a generated `id`, a status typically set to `queued`, and upload metadata. The returned status can change as the Worker runs.

### Check job status

```powershell
curl.exe "http://127.0.0.1:8300/api/v1/transcription/$ID"
```

Optionally poll until processing reaches a terminal state:

```powershell
do {
    Start-Sleep -Seconds 3
    $job = Invoke-RestMethod -Uri "http://127.0.0.1:8300/api/v1/transcription/$ID"
    Write-Host "Status: $($job.status)"
} until ($job.status -in @("completed", "failed", "cancelled"))

$job
```

Supported states are:

```text
created -> uploaded -> queued -> validating -> transcribing
        -> normalizing -> post_processing -> completed

Other terminal states: failed, cancelled
```

### Read the result

After a terminal state has been reached:

```powershell
$result = Invoke-RestMethod -Uri "http://127.0.0.1:8300/api/v1/transcription/$ID/result"

$result.status
$result.raw_text
$result.normalized_text
$result.processed_text
$result.segments
```

For a successful transcription, verify:

- `status` is `completed`.
- `raw_text`, `normalized_text`, and `processed_text` are present.
- `segments` contains timestamps and text.
- `asr_engine` and `asr_model` identify the ASR engine/model (the configured V4 local directory should be recorded as the model).
- With the default **LLM-disabled** configuration, `processed_text` equals `normalized_text`.

If a job fails, check `error_code` and `error_message` in the status/result response and inspect the Worker logs. A result requested before a terminal state may return HTTP `409`.

## 6. Current API endpoints

Base path: `http://127.0.0.1:8300/api/v1/transcription`

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/` | Upload and queue an audio file (`multipart/form-data`, field: `file`) |
| `GET` | `/{transcription_id}` | Read processing status and metadata |
| `GET` | `/{transcription_id}/result` | Read transcript and segments after a terminal state |
| `POST` | `/{transcription_id}/retry` | Retry an eligible failed/uploaded job |
| `DELETE` | `/{transcription_id}` | Delete a job when its current state allows deletion |

OpenAPI documentation is available at `/docs`. The API can report upload/validation errors (for example, HTTP `400`, `413`, and `415`), missing resources (`404`), invalid states (`409`), and storage/queue outages (`503`).

The `retry` and `delete` endpoints are **state-dependent**. Do not assume they are allowed while a job is actively processing.

## 7. Audio limits and output policy

Current development configuration:

- Allowed extensions: `.aac`, `.flac`, `.m4a`, `.mp3`, `.ogg`, `.opus`, `.wav`, `.webm`.
- Maximum upload size: **500 MiB** (configured as `max_file_size_mb: 500`).
- Maximum audio duration: **14,400 seconds / 4 hours**.
- Requested ASR language: Persian (`fa`).

The extension/content type are initially checked on upload; actual media validity is subsequently checked by `ffprobe` in the Worker. A file may be accepted for queueing but later fail media validation.

The three transcript fields have distinct purposes:

| Field | Meaning |
| --- | --- |
| `raw_text` | Original ASR output; preserved rather than overwritten |
| `normalized_text` | Deterministic Persian Unicode/spacing normalization |
| `processed_text` | LLM-constrained formatting, or normalized text when LLM is disabled |

LLM processing, when enabled, is intended to improve punctuation and readability without summarization, translation, or changing factual content. The default local test **does not exercise the LLM**.

## 8. Run automated tests

With dependencies installed, and local infrastructure running when integration tests require it:

```powershell
python -m pytest -v
```

The repository includes tests for health checks, repositories, upload/validation, queue handling, transcription stages, retry/idempotency, runtime configuration, observability, and verification helpers.

**Important:** Having these tests in the repository does not guarantee that every developer's machine will pass them without the required services, dependencies, and configuration.

## 9. Optional local end-to-end verification script

A separate script can verify the running API/Worker and check the persisted transcript, object storage, and a controlled retry scenario:

```powershell
python -m src.infrastructure.scripts.final_verification `
    --audio "benchmarks/asr/audio/sample-001.wav" `
    --api-base-url "http://127.0.0.1:8300" `
    --skip-log-verification
```

This creates:

```text
diagnostics/final_verification_report.json
```

With `--skip-log-verification`, the expected success marker is:

```json
{"overall_status": "passed_with_log_verification_skipped"}
```

That is a **diagnostic pass**, not a full Stage 14 pass. A strict Stage 14 verification also requires API and Worker log files containing the current run's structured events, provided using `--api-log-file` and `--worker-log-file`, without `--skip-log-verification`.

This script performs real uploads, processes audio, and exercises a controlled failure/retry. Run it only against a disposable local environment or with awareness of the test records it creates.

## 10. ASR benchmark status

The repository includes a development benchmark tool and small sample dataset:

```powershell
python -m src.infrastructure.scripts.benchmark_asr `
    --manifest benchmarks/asr/manifest.jsonl `
    --output-dir benchmarks/asr/results `
    --allow-small-dataset
```

The benchmark compares:

- `openai/whisper-large-v3`
- `openai/whisper-large-v3-turbo`
- `nezamisafa/whisper-persian-v4`

The **current committed sample dataset is for smoke/development testing**, not final model selection. The project's target acceptance dataset is at least **400 files and 10 hours** of real Persian audio, with relevant conditions and comparable settings.

The locally configured Persian Whisper V4 is a requested Local evaluation choice, **not a formally selected production winner**. The benchmark compares all three candidates and requires the converted V4 directory described above. It may require substantial disk space and compute resources.

## 11. Troubleshooting

| Symptom | What to check |
| --- | --- |
| Readiness returns `503` | PostgreSQL container health, local port `15432`, and migration/configuration |
| Upload fails with `503` | MinIO availability, bucket initialization, or RabbitMQ connectivity |
| Job remains `queued` | Worker process running, RabbitMQ container healthy, matching queue configuration |
| Worker fails to load ASR model | Confirm that `models/whisper-persian-v4-ct2/model.bin` and `tokenizer.json` exist, and rerun the Step 2.5 model load check |
| Job fails at `validating` | `ffprobe` installation/PATH, supported media format, real audio content, duration limit |
| First transcription is slow | Persian Whisper V4 is a large model; verify conversion and available CPU/RAM |
| Result endpoint returns `409` | Job has not reached a terminal state |
| `processed_text == normalized_text` | Expected when `llm.enabled: false` |
| API connection refused | Confirm Uvicorn is running and use port **8300** |
| Unexpected production configuration errors | Ensure `ENV_TYPE=development` is set in each Python terminal |

For Docker status/logs:

```powershell
docker compose ps
docker compose logs --tail=100 postgres rabbitmq minio minio_init
```

Stop local infrastructure without deleting persisted Docker volumes:

```powershell
docker compose down
```

## 12. Project status and boundaries

### Implemented in the repository

Local infrastructure configuration, migrations, storage upload, asynchronous queue processing, audio validation, Faster Whisper integration, transcript persistence, normalization, optional LLM integration, API endpoints, automated tests, observability, Docker deployment definitions, and end-to-end verification tooling.

### Not yet accepted as production-ready

- Final large-scale Persian ASR benchmark and data-driven final model selection.
- Deployment and GPU validation on the company's target server.
- Connectivity/integration verification against the company's shared vLLM endpoint.
- Strict production end-to-end acceptance.
- Company-specific access control, credential management, network exposure, and operational policy approval.

Do **not** expose the local development API or its example infrastructure credentials to an untrusted network. Enterprise authentication/authorization and deployment hardening must be reviewed before an externally reachable deployment.

The repo contains production-related artifacts, but **this README's supported walkthrough is the Local test path**.

## 13. Main files

```text
src/host/app.py
src/host/res/appsettings.development.yaml
src/application/transcription/transcription_controller.py
src/application/transcription/dtos/transcription_dto.py
src/core/services/transcription/
src/infrastructure/models/
src/infrastructure/repositories/transcription/
src/infrastructure/providers/asr/faster_whisper_provider.py
src/infrastructure/providers/media/ffmpeg_provider.py
src/infrastructure/providers/storage/s3_storage_provider.py
src/infrastructure/providers/queue/celery_queue_provider.py
src/infrastructure/providers/llm/openai_compatible_llm_provider.py
src/infrastructure/tasks/worker.py
src/infrastructure/tasks/transcription_task.py
src/infrastructure/scripts/run_migrations.py
src/infrastructure/scripts/final_verification.py
tests/
benchmarks/asr/
docker-compose.yml
```

## License and ownership

No license file is currently present in the repository. Before importing this code into a company repository, confirm code ownership and the company's licensing and contribution policy.
