from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import json
import math
import re
import statistics
import subprocess
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import psutil

from src.infrastructure.providers.asr.faster_whisper_provider import (
    FasterWhisperProvider,
)
from src.infrastructure.utils.config_reader import (
    ConfigReader,
)


_ARABIC_DIACRITICS_PATTERN = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

_WHITESPACE_PATTERN = re.compile(
    r"\s+"
)

_PUNCTUATION_PATTERN = re.compile(
    r"[^\w\s\u200c]",
    re.UNICODE,
)


@dataclass(
    frozen=True,
    slots=True,
)
class BenchmarkItem:
    item_id: str
    audio_path: Path
    reference_text: str
    duration_seconds: float
    tags: tuple[str, ...]


@dataclass(
    frozen=True,
    slots=True,
)
class BenchmarkModel:
    name: str
    reference: str


@dataclass(
    frozen=True,
    slots=True,
)
class EditMetricTotals:
    edits: int
    reference_units: int

    @property
    def rate(
        self,
    ) -> float:
        if self.reference_units == 0:
            return (
                0.0
                if self.edits == 0
                else 1.0
            )

        return (
            self.edits
            / self.reference_units
        )


class _MemorySampler:
    def __init__(
        self,
        interval_seconds: float = 0.25,
    ):
        self._interval_seconds = (
            interval_seconds
        )

        self._process = psutil.Process()

        self._stop_event = (
            threading.Event()
        )

        self._thread: (
            threading.Thread | None
        ) = None

        self.peak_ram_mb = 0.0

        self.peak_vram_mb: (
            float | None
        ) = None

    def start(
        self,
    ) -> None:
        self._sample_once()

        self._thread = (
            threading.Thread(
                target=self._run,
                daemon=True,
            )
        )

        self._thread.start()

    def stop(
        self,
    ) -> None:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(
                timeout=2.0
            )

        self._sample_once()

    def _run(
        self,
    ) -> None:
        while not self._stop_event.wait(
            self._interval_seconds
        ):
            self._sample_once()

    def _sample_once(
        self,
    ) -> None:
        try:
            ram_mb = (
                self._process
                .memory_info()
                .rss
                / (1024 * 1024)
            )

            self.peak_ram_mb = max(
                self.peak_ram_mb,
                ram_mb,
            )

        except psutil.Error:
            pass

        vram_mb = (
            self._read_process_vram_mb()
        )

        if vram_mb is not None:
            if self.peak_vram_mb is None:
                self.peak_vram_mb = (
                    vram_mb
                )
            else:
                self.peak_vram_mb = max(
                    self.peak_vram_mb,
                    vram_mb,
                )

    def _read_process_vram_mb(
        self,
    ) -> float | None:
        try:
            completed = (
                subprocess.run(
                    [
                        "nvidia-smi",
                        (
                            "--query-compute-apps="
                            "pid,used_memory"
                        ),
                        (
                            "--format="
                            "csv,noheader,nounits"
                        ),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
            )

        except (
            OSError,
            subprocess.SubprocessError,
        ):
            return None

        if completed.returncode != 0:
            return None

        current_pid = (
            self._process.pid
        )

        total_mb = 0.0
        found = False

        for line in (
            completed.stdout
            .splitlines()
        ):
            parts = [
                part.strip()
                for part in line.split(",")
            ]

            if len(parts) != 2:
                continue

            try:
                pid = int(
                    parts[0]
                )

                memory_mb = float(
                    parts[1]
                )

            except ValueError:
                continue

            if pid == current_pid:
                total_mb += memory_mb
                found = True

        return (
            total_mb
            if found
            else None
        )


def _parse_args(
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Faster Whisper "
            "candidates on the same "
            "Persian dataset."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help=(
            "Path to the JSONL "
            "benchmark manifest."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "benchmarks/asr/results"
        ),
        help=(
            "Directory for benchmark "
            "result files."
        ),
    )

    parser.add_argument(
        "--allow-small-dataset",
        action="store_true",
        help=(
            "Allow a development/smoke "
            "benchmark below the final "
            "minimum file-count or "
            "duration requirements."
        ),
    )

    return parser.parse_args()


def _load_manifest(
    manifest_path: Path,
) -> list[BenchmarkItem]:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Benchmark manifest "
            f"'{manifest_path}' "
            f"was not found"
        )

    items: list[
        BenchmarkItem
    ] = []

    seen_ids: set[str] = set()

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as manifest_file:
        for (
            line_number,
            raw_line,
        ) in enumerate(
            manifest_file,
            start=1,
        ):
            if not raw_line.strip():
                continue

            payload = json.loads(
                raw_line
            )

            item_id = str(
                payload["id"]
            ).strip()

            audio_value = str(
                payload["audio_path"]
            ).strip()

            reference_text = str(
                payload["reference_text"]
            ).strip()

            duration_seconds = float(
                payload["duration_seconds"]
            )

            tags_value = payload.get(
                "tags",
                [],
            )

            if not item_id:
                raise ValueError(
                    f"Manifest line "
                    f"{line_number}: "
                    f"id must not be empty"
                )

            if item_id in seen_ids:
                raise ValueError(
                    f"Manifest line "
                    f"{line_number}: "
                    f"duplicate id "
                    f"'{item_id}'"
                )

            if not audio_value:
                raise ValueError(
                    f"Manifest line "
                    f"{line_number}: "
                    f"audio_path must "
                    f"not be empty"
                )

            if not reference_text:
                raise ValueError(
                    f"Manifest line "
                    f"{line_number}: "
                    f"reference_text must "
                    f"not be empty"
                )

            if duration_seconds <= 0:
                raise ValueError(
                    f"Manifest line "
                    f"{line_number}: "
                    f"duration_seconds "
                    f"must be greater "
                    f"than zero"
                )

            if not isinstance(
                tags_value,
                list,
            ):
                raise ValueError(
                    f"Manifest line "
                    f"{line_number}: "
                    f"tags must be a "
                    f"JSON list"
                )

            audio_path = (
                Path(audio_value)
                .expanduser()
            )

            if not audio_path.is_absolute():
                audio_path = (
                    manifest_path.parent
                    / audio_path
                )

            audio_path = (
                audio_path.resolve()
            )

            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Manifest line "
                    f"{line_number}: "
                    f"audio file "
                    f"'{audio_path}' "
                    f"was not found"
                )

            tags = tuple(
                sorted(
                    {
                        str(tag).strip()
                        for tag
                        in tags_value
                        if str(
                            tag
                        ).strip()
                    }
                )
            )

            seen_ids.add(
                item_id
            )

            items.append(
                BenchmarkItem(
                    item_id=item_id,
                    audio_path=(
                        audio_path
                    ),
                    reference_text=(
                        reference_text
                    ),
                    duration_seconds=(
                        duration_seconds
                    ),
                    tags=tags,
                )
            )

    if not items:
        raise ValueError(
            "Benchmark manifest "
            "contains no items"
        )

    return items


def _load_models(
    config_reader: ConfigReader,
) -> list[BenchmarkModel]:
    raw_models = (
        config_reader.get_required(
            "asr.benchmark.models"
        )
    )

    if (
        not isinstance(
            raw_models,
            list,
        )
        or not raw_models
    ):
        raise ValueError(
            "Configuration key "
            "'asr.benchmark.models' "
            "must be a non-empty list"
        )

    models: list[
        BenchmarkModel
    ] = []

    for (
        index,
        raw_model,
    ) in enumerate(
        raw_models
    ):
        if not isinstance(
            raw_model,
            dict,
        ):
            raise ValueError(
                f"asr.benchmark.models"
                f"[{index}] must "
                f"be a mapping"
            )

        name = str(
            raw_model.get(
                "name",
                "",
            )
        ).strip()

        reference = str(
            raw_model.get(
                "reference",
                "",
            )
        ).strip()

        if (
            not name
            or not reference
        ):
            raise ValueError(
                f"asr.benchmark.models"
                f"[{index}] requires "
                f"name and reference"
            )

        models.append(
            BenchmarkModel(
                name=name,
                reference=reference,
            )
        )

    return models


def _validate_dataset_size(
    items: Sequence[
        BenchmarkItem
    ],
    config_reader: ConfigReader,
    allow_small_dataset: bool,
) -> None:
    minimum_files = int(
        config_reader.get_required(
            "asr.benchmark."
            "minimum_files"
        )
    )

    minimum_duration_hours = float(
        config_reader.get_required(
            "asr.benchmark."
            "minimum_duration_hours"
        )
    )

    if (
        minimum_files <= 0
        or minimum_duration_hours <= 0
    ):
        raise ValueError(
            "ASR benchmark minimums "
            "must be greater than zero"
        )

    if allow_small_dataset:
        return

    total_duration_hours = (
        sum(
            item.duration_seconds
            for item in items
        )
        / 3600
    )

    failures = []

    if len(items) < minimum_files:
        failures.append(
            f"{len(items)} files < "
            f"required {minimum_files}"
        )

    if (
        total_duration_hours
        < minimum_duration_hours
    ):
        failures.append(
            f"{total_duration_hours:.2f} "
            f"hours < required "
            f"{minimum_duration_hours:.2f} "
            f"hours"
        )

    if failures:
        raise ValueError(
            "Benchmark dataset does not "
            "satisfy final acceptance "
            "minimums: "
            + "; ".join(
                failures
            )
            + ". Use --allow-small-dataset "
            "only for development "
            "smoke runs."
        )


def _normalize_for_metric(
    text: str,
) -> str:
    normalized = (
        unicodedata.normalize(
            "NFKC",
            text,
        )
    )

    normalized = (
        normalized.replace(
            "ي",
            "ی",
        )
    )

    normalized = (
        normalized.replace(
            "ى",
            "ی",
        )
    )

    normalized = (
        normalized.replace(
            "ك",
            "ک",
        )
    )

    normalized = (
        normalized.replace(
            "ۀ",
            "هٔ",
        )
    )

    normalized = (
        normalized.replace(
            "ة",
            "ه",
        )
    )

    normalized = (
        normalized.replace(
            "ؤ",
            "و",
        )
    )

    normalized = (
        normalized.replace(
            "إ",
            "ا",
        )
    )

    normalized = (
        normalized.replace(
            "أ",
            "ا",
        )
    )

    normalized = (
        normalized.replace(
            "ٱ",
            "ا",
        )
    )

    normalized = (
        normalized.replace(
            "ـ",
            "",
        )
    )

    normalized = (
        _ARABIC_DIACRITICS_PATTERN
        .sub(
            "",
            normalized,
        )
    )

    normalized = (
        _PUNCTUATION_PATTERN
        .sub(
            " ",
            normalized,
        )
    )

    normalized = (
        _WHITESPACE_PATTERN
        .sub(
            " ",
            normalized,
        )
    )

    return (
        normalized
        .strip()
        .lower()
    )


def _collapse_whitespace(
    text: str,
) -> str:
    return (
        _WHITESPACE_PATTERN
        .sub(
            " ",
            text,
        )
        .strip()
    )


def _levenshtein_distance(
    reference: Sequence[str],
    hypothesis: Sequence[str],
) -> int:
    if (
        len(reference)
        < len(hypothesis)
    ):
        (
            reference,
            hypothesis,
        ) = (
            hypothesis,
            reference,
        )

    previous_row = list(
        range(
            len(hypothesis) + 1
        )
    )

    for (
        reference_index,
        reference_unit,
    ) in enumerate(
        reference,
        start=1,
    ):
        current_row = [
            reference_index
        ]

        for (
            hypothesis_index,
            hypothesis_unit,
        ) in enumerate(
            hypothesis,
            start=1,
        ):
            insertion = (
                current_row[
                    hypothesis_index - 1
                ]
                + 1
            )

            deletion = (
                previous_row[
                    hypothesis_index
                ]
                + 1
            )

            substitution = (
                previous_row[
                    hypothesis_index - 1
                ]
                + (
                    reference_unit
                    != hypothesis_unit
                )
            )

            current_row.append(
                min(
                    insertion,
                    deletion,
                    substitution,
                )
            )

        previous_row = (
            current_row
        )

    return previous_row[-1]


def _word_metric_totals(
    reference_text: str,
    hypothesis_text: str,
    *,
    normalized: bool,
) -> EditMetricTotals:
    reference = (
        _normalize_for_metric(
            reference_text
        )
        if normalized
        else _collapse_whitespace(
            reference_text
        )
    )

    hypothesis = (
        _normalize_for_metric(
            hypothesis_text
        )
        if normalized
        else _collapse_whitespace(
            hypothesis_text
        )
    )

    reference_units = (
        reference.split()
        if reference
        else []
    )

    hypothesis_units = (
        hypothesis.split()
        if hypothesis
        else []
    )

    return EditMetricTotals(
        edits=(
            _levenshtein_distance(
                reference_units,
                hypothesis_units,
            )
        ),
        reference_units=len(
            reference_units
        ),
    )


def _character_metric_totals(
    reference_text: str,
    hypothesis_text: str,
    *,
    normalized: bool,
) -> EditMetricTotals:
    reference = (
        _normalize_for_metric(
            reference_text
        )
        if normalized
        else _collapse_whitespace(
            reference_text
        )
    )

    hypothesis = (
        _normalize_for_metric(
            hypothesis_text
        )
        if normalized
        else _collapse_whitespace(
            hypothesis_text
        )
    )

    reference_units = list(
        reference
    )

    hypothesis_units = list(
        hypothesis
    )

    return EditMetricTotals(
        edits=(
            _levenshtein_distance(
                reference_units,
                hypothesis_units,
            )
        ),
        reference_units=len(
            reference_units
        ),
    )


def _percentile(
    values: Sequence[float],
    percentile: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(
        values
    )

    if len(ordered) == 1:
        return ordered[0]

    position = (
        (len(ordered) - 1)
        * percentile
    )

    lower_index = math.floor(
        position
    )

    upper_index = math.ceil(
        position
    )

    if (
        lower_index
        == upper_index
    ):
        return ordered[
            lower_index
        ]

    fraction = (
        position
        - lower_index
    )

    return (
        ordered[
            lower_index
        ]
        * (1 - fraction)
        + ordered[
            upper_index
        ]
        * fraction
    )


def _add_totals(
    left: EditMetricTotals,
    right: EditMetricTotals,
) -> EditMetricTotals:
    return EditMetricTotals(
        edits=(
            left.edits
            + right.edits
        ),
        reference_units=(
            left.reference_units
            + right.reference_units
        ),
    )


def _empty_totals(
) -> EditMetricTotals:
    return EditMetricTotals(
        edits=0,
        reference_units=0,
    )


async def _run_model_benchmark(
    provider: FasterWhisperProvider,
    model: BenchmarkModel,
    items: Sequence[
        BenchmarkItem
    ],
    output_file,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
]:
    memory_sampler = (
        _MemorySampler()
    )

    memory_sampler.start()

    load_started = (
        time.perf_counter()
    )

    try:
        await provider.prepare_model_async(
            model.reference
        )

    except Exception:
        memory_sampler.stop()
        raise

    model_load_seconds = (
        time.perf_counter()
        - load_started
    )

    word_totals = (
        _empty_totals()
    )

    char_totals = (
        _empty_totals()
    )

    normalized_word_totals = (
        _empty_totals()
    )

    normalized_char_totals = (
        _empty_totals()
    )

    tag_totals: dict[
        str,
        dict[str, object],
    ] = {}

    latencies: list[
        float
    ] = []

    total_inference_seconds = 0.0
    total_audio_seconds = 0.0
    error_count = 0

    try:
        for item in items:
            inference_started = (
                time.perf_counter()
            )

            try:
                result = (
                    await provider
                    .transcribe_async(
                        item.audio_path,
                        model_reference=(
                            model.reference
                        ),
                    )
                )

                inference_seconds = (
                    time.perf_counter()
                    - inference_started
                )

                hypothesis_text = (
                    result.text
                )

                detected_language = (
                    result
                    .detected_language
                )

                language_probability = (
                    result
                    .language_probability
                )

                segment_count = len(
                    result.segments
                )

                asr_duration_seconds = (
                    result
                    .duration_seconds
                )

                error_message = None

            except Exception as exc:
                inference_seconds = (
                    time.perf_counter()
                    - inference_started
                )

                hypothesis_text = ""
                detected_language = ""
                language_probability = 0.0
                segment_count = 0
                asr_duration_seconds = 0.0

                error_message = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                error_count += 1

            latencies.append(
                inference_seconds
            )

            total_inference_seconds += (
                inference_seconds
            )

            total_audio_seconds += (
                item.duration_seconds
            )

            raw_word = (
                _word_metric_totals(
                    item.reference_text,
                    hypothesis_text,
                    normalized=False,
                )
            )

            raw_char = (
                _character_metric_totals(
                    item.reference_text,
                    hypothesis_text,
                    normalized=False,
                )
            )

            normalized_word = (
                _word_metric_totals(
                    item.reference_text,
                    hypothesis_text,
                    normalized=True,
                )
            )

            normalized_char = (
                _character_metric_totals(
                    item.reference_text,
                    hypothesis_text,
                    normalized=True,
                )
            )

            word_totals = _add_totals(
                word_totals,
                raw_word,
            )

            char_totals = _add_totals(
                char_totals,
                raw_char,
            )

            normalized_word_totals = (
                _add_totals(
                    normalized_word_totals,
                    normalized_word,
                )
            )

            normalized_char_totals = (
                _add_totals(
                    normalized_char_totals,
                    normalized_char,
                )
            )

            for tag in item.tags:
                tag_state = (
                    tag_totals.setdefault(
                        tag,
                        {
                            "files": 0,
                            "audio_seconds": 0.0,
                            "word": (
                                _empty_totals()
                            ),
                            "char": (
                                _empty_totals()
                            ),
                            "normalized_word": (
                                _empty_totals()
                            ),
                            "normalized_char": (
                                _empty_totals()
                            ),
                        },
                    )
                )

                tag_state[
                    "files"
                ] = (
                    int(
                        tag_state[
                            "files"
                        ]
                    )
                    + 1
                )

                tag_state[
                    "audio_seconds"
                ] = (
                    float(
                        tag_state[
                            "audio_seconds"
                        ]
                    )
                    + item.duration_seconds
                )

                tag_state[
                    "word"
                ] = _add_totals(
                    tag_state[
                        "word"
                    ],
                    raw_word,
                )

                tag_state[
                    "char"
                ] = _add_totals(
                    tag_state[
                        "char"
                    ],
                    raw_char,
                )

                tag_state[
                    "normalized_word"
                ] = _add_totals(
                    tag_state[
                        "normalized_word"
                    ],
                    normalized_word,
                )

                tag_state[
                    "normalized_char"
                ] = _add_totals(
                    tag_state[
                        "normalized_char"
                    ],
                    normalized_char,
                )

            output_file.write(
                json.dumps(
                    {
                        "model_name": (
                            model.name
                        ),
                        "model_reference": (
                            model.reference
                        ),
                        "id": (
                            item.item_id
                        ),
                        "audio_path": str(
                            item.audio_path
                        ),
                        "tags": list(
                            item.tags
                        ),
                        "reference_text": (
                            item
                            .reference_text
                        ),
                        "hypothesis_text": (
                            hypothesis_text
                        ),
                        (
                            "manifest_"
                            "duration_seconds"
                        ): (
                            item
                            .duration_seconds
                        ),
                        (
                            "asr_"
                            "duration_seconds"
                        ): (
                            asr_duration_seconds
                        ),
                        "inference_seconds": (
                            inference_seconds
                        ),
                        "rtf": (
                            inference_seconds
                            / item.duration_seconds
                        ),
                        (
                            "detected_language"
                        ): (
                            detected_language
                        ),
                        (
                            "language_probability"
                        ): (
                            language_probability
                        ),
                        "segment_count": (
                            segment_count
                        ),
                        "wer": (
                            raw_word.rate
                        ),
                        "cer": (
                            raw_char.rate
                        ),
                        "normalized_wer": (
                            normalized_word.rate
                        ),
                        "normalized_cer": (
                            normalized_char.rate
                        ),
                        "error": (
                            error_message
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            output_file.flush()

    finally:
        memory_sampler.stop()

    summary = {
        "model_name": (
            model.name
        ),
        "model_reference": (
            model.reference
        ),
        "files": len(
            items
        ),
        "errors": (
            error_count
        ),
        "audio_hours": (
            total_audio_seconds
            / 3600
        ),
        "model_load_seconds": (
            model_load_seconds
        ),
        "total_inference_seconds": (
            total_inference_seconds
        ),
        "rtf": (
            total_inference_seconds
            / total_audio_seconds
            if total_audio_seconds > 0
            else 0.0
        ),
        "avg_latency_seconds": (
            statistics.fmean(
                latencies
            )
            if latencies
            else 0.0
        ),
        "p50_latency_seconds": (
            _percentile(
                latencies,
                0.50,
            )
        ),
        "p95_latency_seconds": (
            _percentile(
                latencies,
                0.95,
            )
        ),
        "wer": (
            word_totals.rate
        ),
        "cer": (
            char_totals.rate
        ),
        "normalized_wer": (
            normalized_word_totals
            .rate
        ),
        "normalized_cer": (
            normalized_char_totals
            .rate
        ),
        "peak_ram_mb": (
            memory_sampler
            .peak_ram_mb
        ),
        "peak_vram_mb": (
            memory_sampler
            .peak_vram_mb
        ),
    }

    tag_rows: list[
        dict[str, object]
    ] = []

    for (
        tag,
        state,
    ) in sorted(
        tag_totals.items()
    ):
        tag_rows.append(
            {
                "model_name": (
                    model.name
                ),
                "model_reference": (
                    model.reference
                ),
                "tag": tag,
                "files": int(
                    state["files"]
                ),
                "audio_hours": (
                    float(
                        state[
                            "audio_seconds"
                        ]
                    )
                    / 3600
                ),
                "wer": (
                    state[
                        "word"
                    ].rate
                ),
                "cer": (
                    state[
                        "char"
                    ].rate
                ),
                "normalized_wer": (
                    state[
                        "normalized_word"
                    ].rate
                ),
                "normalized_cer": (
                    state[
                        "normalized_char"
                    ].rate
                ),
            }
        )

    return (
        summary,
        tag_rows,
    )


def _write_csv(
    path: Path,
    rows: Sequence[
        dict[str, object]
    ],
) -> None:
    if not rows:
        return

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def _print_summary(
    rows: Sequence[
        dict[str, object]
    ],
) -> None:
    headers = [
        "model_name",
        "wer",
        "cer",
        "normalized_wer",
        "normalized_cer",
        "rtf",
        "p95_latency_seconds",
        "peak_ram_mb",
        "peak_vram_mb",
        "errors",
    ]

    print(
        "\nASR benchmark summary"
    )

    print(
        " | ".join(
            headers
        )
    )

    print(
        "-" * 160
    )

    for row in rows:
        values = []

        for header in headers:
            value = row[
                header
            ]

            if isinstance(
                value,
                float,
            ):
                values.append(
                    f"{value:.4f}"
                )

            elif value is None:
                values.append(
                    "n/a"
                )

            else:
                values.append(
                    str(value)
                )

        print(
            " | ".join(
                values
            )
        )


def _validate_required_tags(
    items: Sequence[
        BenchmarkItem
    ],
    config_reader: ConfigReader,
) -> None:
    required_tags = (
        config_reader.get(
            "asr.benchmark."
            "required_tags",
            [],
        )
    )

    if not isinstance(
        required_tags,
        list,
    ):
        raise ValueError(
            "Configuration key "
            "'asr.benchmark."
            "required_tags' must "
            "be a list"
        )

    present_tags = {
        tag
        for item in items
        for tag in item.tags
    }

    missing_tags = [
        str(tag)
        for tag in required_tags
        if str(tag)
        not in present_tags
    ]

    if missing_tags:
        raise ValueError(
            "Benchmark dataset is "
            "missing required coverage "
            "tags: "
            + ", ".join(
                missing_tags
            )
        )


async def _main_async(
    args: argparse.Namespace,
) -> None:
    config_reader = (
        ConfigReader()
    )

    manifest_path = (
        Path(
            args.manifest
        )
        .expanduser()
        .resolve()
    )

    output_dir = (
        Path(
            args.output_dir
        )
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    items = _load_manifest(
        manifest_path
    )

    models = _load_models(
        config_reader
    )

    _validate_dataset_size(
        items,
        config_reader,
        args.allow_small_dataset,
    )

    _validate_required_tags(
        items,
        config_reader,
    )

    provider = (
        FasterWhisperProvider(
            config_reader=(
                config_reader
            )
        )
    )

    summary_rows: list[
        dict[str, object]
    ] = []

    tag_rows: list[
        dict[str, object]
    ] = []

    per_file_path = (
        output_dir
        / "per_file_results.jsonl"
    )

    with per_file_path.open(
        "w",
        encoding="utf-8",
    ) as per_file_output:
        for model in models:
            print(
                f"Benchmarking "
                f"{model.name} "
                f"({model.reference}) ..."
            )

            try:
                (
                    summary,
                    model_tag_rows,
                ) = (
                    await _run_model_benchmark(
                        provider=provider,
                        model=model,
                        items=items,
                        output_file=(
                            per_file_output
                        ),
                    )
                )

                summary_rows.append(
                    summary
                )

                tag_rows.extend(
                    model_tag_rows
                )

            finally:
                await (
                    provider
                    .unload_model_async(
                        model.reference
                    )
                )

                gc.collect()

    _write_csv(
        output_dir
        / "summary.csv",
        summary_rows,
    )

    _write_csv(
        output_dir
        / "summary_by_tag.csv",
        tag_rows,
    )

    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            summary_rows,
            summary_file,
            ensure_ascii=False,
            indent=2,
        )

    _print_summary(
        summary_rows
    )

    print(
        "\nNo winner is selected "
        "automatically. "
        "Choose Local/Production "
        "models only after reviewing "
        "accuracy, RTF, latency, "
        "RAM/VRAM, failures, and "
        "tag coverage."
    )


def main(
) -> None:
    args = _parse_args()

    asyncio.run(
        _main_async(
            args
        )
    )


if __name__ == "__main__":
    main()