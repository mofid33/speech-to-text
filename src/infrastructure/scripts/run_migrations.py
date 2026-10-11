from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config


PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.infrastructure.utils.config_reader import (  # noqa: E402
    ConfigReader,
)


ALEMBIC_DIRECTORY = (
    PROJECT_ROOT
    / "src"
    / "infrastructure"
    / "alembic"
)


def _normalize_database_url(
    database_url: str,
) -> str:
    if database_url.startswith(
        "postgresql://"
    ):
        return database_url.replace(
            "postgresql://",
            "postgresql+asyncpg://",
            1,
        )

    return database_url


def _build_alembic_config() -> Config:
    config_reader = ConfigReader()

    database_url = config_reader.get(
        "database.url"
    )

    if (
        not database_url
        or not isinstance(
            database_url,
            str,
        )
    ):
        raise RuntimeError(
            "Configuration key "
            "'database.url' "
            "is missing or invalid"
        )

    alembic_config = Config()

    alembic_config.set_main_option(
        "script_location",
        str(ALEMBIC_DIRECTORY),
    )

    alembic_config.set_main_option(
        "sqlalchemy.url",
        _normalize_database_url(
            database_url
        ).replace(
            "%",
            "%%",
        ),
    )

    return alembic_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manage database migrations"
        )
    )

    action_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    action_group.add_argument(
        "--upgrade",
        action="store_true",
        help=(
            "Upgrade the database to "
            "the latest committed migration"
        ),
    )

    action_group.add_argument(
        "--autogenerate",
        action="store_true",
        help=(
            "Generate a new migration from "
            "SQLAlchemy metadata changes"
        ),
    )

    action_group.add_argument(
        "--sync",
        action="store_true",
        help=(
            "Autogenerate a migration and "
            "immediately upgrade to head"
        ),
    )

    parser.add_argument(
        "-m",
        "--message",
        type=str,
        help=(
            "Migration message for "
            "--autogenerate or --sync"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    alembic_config = (
        _build_alembic_config()
    )

    if args.upgrade:
        command.upgrade(
            alembic_config,
            "head",
        )
        return

    if (
        not args.message
        or not args.message.strip()
    ):
        raise SystemExit(
            "-m/--message is required "
            "for --autogenerate and --sync"
        )

    command.revision(
        alembic_config,
        message=args.message.strip(),
        autogenerate=True,
    )

    if args.sync:
        command.upgrade(
            alembic_config,
            "head",
        )


if __name__ == "__main__":
    main()