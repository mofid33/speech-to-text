from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import ModuleType

from src.infrastructure.di.inject import register_provider

_EXCLUDED_PACKAGE_PARTS = {
    "__pycache__",
    "alembic",
    "di",
    "dtos",
    "models",
    "res",
    "scripts",
    "versions",
}


def _should_skip_path(relative_path: Path) -> bool:
    return bool(
        set(relative_path.parts).intersection(_EXCLUDED_PACKAGE_PARTS)
    )


def _register_module_providers(module: ModuleType) -> None:
    for _, candidate in inspect.getmembers(module, inspect.isclass):
        if candidate.__module__ != module.__name__:
            continue

        if getattr(candidate, "__di_provider__", False):
            register_provider(candidate)


def bootstrap_package(package_name: str) -> None:
    package = importlib.import_module(package_name)
    package_paths = getattr(package, "__path__", None)

    if package_paths is None:
        return

    for package_path in package_paths:
        root_path = Path(package_path)

        for module_file in sorted(root_path.rglob("*.py")):
            relative_path = module_file.relative_to(root_path)

            if _should_skip_path(relative_path):
                continue

            module_parts = relative_path.with_suffix("").parts

            if module_parts[-1] == "__init__":
                module_parts = module_parts[:-1]

            if not module_parts:
                continue

            module_name = ".".join(
                (package_name, *module_parts)
            )

            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue

            _register_module_providers(module)