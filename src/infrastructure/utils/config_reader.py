from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml

from src.infrastructure.di.inject import inject

T = TypeVar("T")

_ENVIRONMENT_PLACEHOLDER_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^{}]*))?\}"
)


@inject
class ConfigReader:
    __di_singleton__ = True

    def __init__(self) -> None:
        self._environment = (
            os.getenv("env_type")
            or os.getenv("ENV_TYPE")
            or "development"
        ).strip().lower()

        resources_directory = (
            Path(__file__).resolve().parents[2]
            / "host"
            / "res"
        )

        config_filename = (
            "appsettings.development.yaml"
            if self._environment == "development"
            else "appsettings.yaml"
        )

        self._config_path = (
            resources_directory
            / config_filename
        )

        if not self._config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file "
                f"'{self._config_path}' was not found."
            )

        with self._config_path.open(
            "r",
            encoding="utf-8",
        ) as config_file:
            loaded_config = (
                yaml.safe_load(config_file)
                or {}
            )

        if not isinstance(loaded_config, dict):
            raise ValueError(
                f"Configuration file "
                f"'{self._config_path}' must contain "
                "a YAML mapping."
            )

        self._config: dict[str, Any] = loaded_config

    def get(
        self,
        key: str,
        default: T | None = None,
    ) -> Any | T | None:
        current: Any = self._config

        for part in key.split("."):
            if (
                not isinstance(current, dict)
                or part not in current
            ):
                return self._resolve_environment_placeholders(
                    default,
                    key=key,
                )

            current = current[part]

        return self._resolve_environment_placeholders(
            current,
            key=key,
        )

    def get_required(self, key: str) -> Any:
        value = self.get(key)

        if value is None:
            raise KeyError(
                f"Required configuration key "
                f"'{key}' is missing in "
                f"'{self._config_path}'."
            )

        return value

    def get_environment(self) -> str:
        return self._environment

    def get_app_name(self) -> str:
        return cast(
            str,
            self.get_required("app.name"),
        )

    def get_app_version(self) -> str:
        return cast(
            str,
            self.get_required("app.version"),
        )

    def get_api_prefix(self) -> str:
        prefix = cast(
            str,
            self.get_required("api.prefix"),
        ).strip()

        if not prefix.startswith("/"):
            prefix = f"/{prefix}"

        return prefix.rstrip("/")

    def get_server_host(self) -> str:
        return cast(
            str,
            self.get_required("server.host"),
        )

    def get_server_port(self) -> int:
        return int(
            self.get_required("server.port")
        )

    def get_database_url(self) -> str:
        return cast(
            str,
            self.get_required("database.url"),
        )

    @staticmethod
    def _resolve_environment_placeholders(
        value: Any,
        *,
        key: str,
    ) -> Any:
        if (
            not isinstance(value, str)
            or "${" not in value
        ):
            return value

        def replace_placeholder(
            match: re.Match[str],
        ) -> str:
            environment_name = match.group(1)
            default_value = match.group(2)

            environment_value = os.getenv(
                environment_name
            )

            if environment_value is not None:
                return environment_value

            if default_value is not None:
                return default_value

            raise RuntimeError(
                "Required environment variable "
                f"'{environment_name}' for "
                f"configuration key '{key}' "
                "is not set."
            )

        return _ENVIRONMENT_PLACEHOLDER_PATTERN.sub(
            replace_placeholder,
            value,
        )