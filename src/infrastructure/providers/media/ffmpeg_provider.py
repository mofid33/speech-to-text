import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.infrastructure.di.inject import inject
from src.infrastructure.utils.config_reader import ConfigReader


class MediaProbeError(RuntimeError):
    """Raised when ffprobe cannot complete reliably."""


class InvalidMediaError(ValueError):
    """Raised when the probed file is not valid audio media."""


@dataclass(frozen=True, slots=True)
class AudioMediaInfo:
    duration_seconds: float
    codec_name: str
    sample_rate_hz: int
    channels: int


@inject
class FfmpegProvider:
    __di_singleton__ = True

    def __init__(self, config_reader: ConfigReader):
        self._ffprobe_path = str(
            config_reader.get_required("media.ffprobe_path")
        ).strip()
        self._probe_timeout_seconds = self._read_positive_float(
            config_reader=config_reader,
            key="media.ffprobe_timeout_seconds",
        )

        if not self._ffprobe_path:
            raise ValueError("media.ffprobe_path must not be empty")

    async def probe_audio_async(
        self,
        file_path: str | Path,
    ) -> AudioMediaInfo:
        normalized_path = Path(file_path)

        if not normalized_path.is_file():
            raise MediaProbeError(
                f"Media file '{normalized_path}' does not exist"
            )

        command = (
            self._ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            (
                "stream=codec_type,codec_name,sample_rate,channels,duration:"
                "stream_disposition=attached_pic:format=duration"
            ),
            "-of",
            "json",
            str(normalized_path),
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, OSError) as exc:
            raise MediaProbeError(
                "ffprobe could not be started"
            ) from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self._probe_timeout_seconds,
            )
        except TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise MediaProbeError(
                "ffprobe timed out while validating audio"
            ) from exc

        if process.returncode != 0:
            error_text = stderr.decode(
                "utf-8",
                errors="replace",
            ).strip()

            message = "Uploaded file is not valid media"
            if error_text:
                message = f"{message}: {error_text}"

            raise InvalidMediaError(message)

        try:
            payload = json.loads(
                stdout.decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MediaProbeError(
                "ffprobe returned an unreadable response"
            ) from exc

        if not isinstance(payload, dict):
            raise MediaProbeError(
                "ffprobe returned an unexpected response"
            )

        return self._parse_audio_info(payload)

    def _parse_audio_info(
        self,
        payload: dict[str, Any],
    ) -> AudioMediaInfo:
        raw_streams = payload.get("streams", [])

        if not isinstance(raw_streams, list):
            raise MediaProbeError(
                "ffprobe response does not contain a stream list"
            )

        streams = [
            stream
            for stream in raw_streams
            if isinstance(stream, dict)
        ]

        audio_streams = [
            stream
            for stream in streams
            if stream.get("codec_type") == "audio"
        ]

        if not audio_streams:
            raise InvalidMediaError(
                "Uploaded file does not contain an audio stream"
            )

        if self._contains_real_video_stream(streams):
            raise InvalidMediaError(
                "Video files are not supported"
            )

        audio_stream = audio_streams[0]

        codec_name = self._required_string(
            audio_stream.get("codec_name"),
            field_name="codec_name",
        )
        sample_rate_hz = self._required_positive_int(
            audio_stream.get("sample_rate"),
            field_name="sample_rate",
        )
        channels = self._required_positive_int(
            audio_stream.get("channels"),
            field_name="channels",
        )

        duration_seconds = self._read_duration_seconds(
            audio_stream=audio_stream,
            format_info=payload.get("format"),
        )

        return AudioMediaInfo(
            duration_seconds=duration_seconds,
            codec_name=codec_name,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
        )

    @staticmethod
    def _contains_real_video_stream(
        streams: list[dict[str, Any]],
    ) -> bool:
        for stream in streams:
            if stream.get("codec_type") != "video":
                continue

            disposition = stream.get("disposition")

            if isinstance(disposition, dict):
                attached_pic = disposition.get("attached_pic")

                try:
                    if int(attached_pic or 0) == 1:
                        continue
                except (TypeError, ValueError):
                    pass

            return True

        return False

    def _read_duration_seconds(
        self,
        *,
        audio_stream: dict[str, Any],
        format_info: Any,
    ) -> float:
        candidates = [audio_stream.get("duration")]

        if isinstance(format_info, dict):
            candidates.append(format_info.get("duration"))

        for candidate in candidates:
            try:
                duration = float(candidate)
            except (TypeError, ValueError):
                continue

            if duration > 0:
                return duration

        raise InvalidMediaError(
            "Audio duration could not be determined"
        )

    @staticmethod
    def _required_string(
        value: Any,
        *,
        field_name: str,
    ) -> str:
        if value is None:
            raise InvalidMediaError(
                f"Audio {field_name} is missing"
            )

        normalized = str(value).strip()

        if not normalized:
            raise InvalidMediaError(
                f"Audio {field_name} is missing"
            )

        return normalized

    @staticmethod
    def _required_positive_int(
        value: Any,
        *,
        field_name: str,
    ) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise InvalidMediaError(
                f"Audio {field_name} is invalid"
            ) from exc

        if parsed <= 0:
            raise InvalidMediaError(
                f"Audio {field_name} is invalid"
            )

        return parsed

    @staticmethod
    def _read_positive_float(
        *,
        config_reader: ConfigReader,
        key: str,
    ) -> float:
        raw_value = config_reader.get_required(key)

        if isinstance(raw_value, bool):
            raise ValueError(
                f"Configuration key '{key}' must be a positive number"
            )

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Configuration key '{key}' must be a positive number"
            ) from exc

        if value <= 0:
            raise ValueError(
                f"Configuration key '{key}' must be a positive number"
            )

        return value