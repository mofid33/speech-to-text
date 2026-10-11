import asyncio
from pathlib import Path
from typing import BinaryIO

import boto3
from boto3.exceptions import Boto3Error
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from src.infrastructure.di.inject import inject
from src.infrastructure.utils.config_reader import ConfigReader


class StorageProviderError(RuntimeError):
    """Raised when an object-storage operation fails."""


@inject
class S3StorageProvider:
    __di_singleton__ = True

    def __init__(self, config_reader: ConfigReader):
        self._bucket_name = str(
            config_reader.get_required("storage.bucket_name")
        ).strip()

        endpoint_url = self._optional_string(
            config_reader.get("storage.endpoint_url")
        )
        region_name = self._optional_string(
            config_reader.get("storage.region_name")
        )
        access_key_id = self._optional_string(
            config_reader.get("storage.access_key_id")
        )
        secret_access_key = self._optional_string(
            config_reader.get("storage.secret_access_key")
        )

        addressing_style = str(
            config_reader.get_required("storage.addressing_style")
        ).strip()

        self._auto_create_bucket = self._as_bool(
            config_reader.get_required("storage.auto_create_bucket")
        )
        verify_ssl = self._as_bool(
            config_reader.get_required("storage.verify_ssl")
        )

        if not self._bucket_name:
            raise ValueError("storage.bucket_name must not be empty")

        if addressing_style not in {"auto", "path", "virtual"}:
            raise ValueError(
                "storage.addressing_style must be one of: "
                "'auto', 'path', 'virtual'"
            )

        client_kwargs = {
            "service_name": "s3",
            "endpoint_url": endpoint_url,
            "region_name": region_name,
            "verify": verify_ssl,
            "config": BotoConfig(
                s3={"addressing_style": addressing_style},
                retries={
                    "max_attempts": 3,
                    "mode": "standard",
                },
            ),
        }

        if access_key_id is not None:
            client_kwargs["aws_access_key_id"] = access_key_id

        if secret_access_key is not None:
            client_kwargs["aws_secret_access_key"] = secret_access_key

        self._client = boto3.client(**client_kwargs)

        self._region_name = region_name
        self._bucket_ready = False
        self._bucket_lock = asyncio.Lock()

    async def upload_file_async(
        self,
        file_obj: BinaryIO,
        object_key: str,
        content_type: str | None = None,
    ) -> str:
        if not object_key:
            raise ValueError("object_key must not be empty")

        if self._auto_create_bucket:
            await self._ensure_bucket_exists_async()

        extra_args: dict[str, str] = {}

        if content_type:
            extra_args["ContentType"] = content_type

        try:
            await asyncio.to_thread(file_obj.seek, 0)

            if extra_args:
                await asyncio.to_thread(
                    self._client.upload_fileobj,
                    file_obj,
                    self._bucket_name,
                    object_key,
                    ExtraArgs=extra_args,
                )
            else:
                await asyncio.to_thread(
                    self._client.upload_fileobj,
                    file_obj,
                    self._bucket_name,
                    object_key,
                )

            return object_key

        except (
            Boto3Error,
            BotoCoreError,
            ClientError,
            OSError,
        ) as exc:
            raise StorageProviderError(
                f"Failed to upload object '{object_key}'"
            ) from exc

    async def download_file_async(
        self,
        object_key: str,
        destination_path: str | Path,
    ) -> Path:
        if not object_key:
            raise ValueError("object_key must not be empty")

        normalized_destination = Path(destination_path)

        try:
            await asyncio.to_thread(
                normalized_destination.parent.mkdir,
                parents=True,
                exist_ok=True,
            )

            await asyncio.to_thread(
                self._client.download_file,
                self._bucket_name,
                object_key,
                str(normalized_destination),
            )

            return normalized_destination

        except (
            Boto3Error,
            BotoCoreError,
            ClientError,
            OSError,
        ) as exc:
            raise StorageProviderError(
                f"Failed to download object '{object_key}'"
            ) from exc

    async def delete_file_async(self, object_key: str) -> None:
        if not object_key:
            return

        try:
            await asyncio.to_thread(
                self._client.delete_object,
                Bucket=self._bucket_name,
                Key=object_key,
            )

        except (
            Boto3Error,
            BotoCoreError,
            ClientError,
            OSError,
        ) as exc:
            raise StorageProviderError(
                f"Failed to delete object '{object_key}'"
            ) from exc

    async def _ensure_bucket_exists_async(self) -> None:
        if self._bucket_ready:
            return

        async with self._bucket_lock:
            if self._bucket_ready:
                return

            try:
                await asyncio.to_thread(
                    self._client.head_bucket,
                    Bucket=self._bucket_name,
                )

                self._bucket_ready = True
                return

            except ClientError as exc:
                error_code = str(
                    exc.response.get("Error", {}).get("Code", "")
                )
                status_code = exc.response.get(
                    "ResponseMetadata", {}
                ).get("HTTPStatusCode")

                bucket_not_found = (
                    status_code == 404
                    or error_code
                    in {
                        "404",
                        "NoSuchBucket",
                        "NotFound",
                    }
                )

                if not bucket_not_found:
                    raise StorageProviderError(
                        f"Failed to access bucket "
                        f"'{self._bucket_name}'"
                    ) from exc

            except (
                Boto3Error,
                BotoCoreError,
                OSError,
            ) as exc:
                raise StorageProviderError(
                    f"Failed to access bucket "
                    f"'{self._bucket_name}'"
                ) from exc

            try:
                create_kwargs = {
                    "Bucket": self._bucket_name,
                }

                if (
                    self._region_name
                    and self._region_name != "us-east-1"
                ):
                    create_kwargs["CreateBucketConfiguration"] = {
                        "LocationConstraint": self._region_name,
                    }

                await asyncio.to_thread(
                    self._client.create_bucket,
                    **create_kwargs,
                )

                self._bucket_ready = True

            except (
                Boto3Error,
                BotoCoreError,
                ClientError,
                OSError,
            ) as exc:
                raise StorageProviderError(
                    f"Failed to create bucket "
                    f"'{self._bucket_name}'"
                ) from exc

    @staticmethod
    def _optional_string(value: object) -> str | None:
        if value is None:
            return None

        normalized = str(value).strip()

        if not normalized:
            return None

        return normalized

    @staticmethod
    def _as_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value

        normalized = str(value).strip().lower()

        if normalized in {"1", "true", "yes", "on"}:
            return True

        if normalized in {"0", "false", "no", "off"}:
            return False

        raise ValueError(
            f"Cannot convert configuration value '{value}' to bool"
        )