from __future__ import annotations

from threading import Lock

from src.infrastructure.di.reflection import bootstrap_package

_bootstrap_lock = Lock()
_is_bootstrapped = False


def bootstrap_di() -> None:
    global _is_bootstrapped

    if _is_bootstrapped:
        return

    with _bootstrap_lock:
        if _is_bootstrapped:
            return

        for package_name in (
            "src.infrastructure",
            "src.core",
            "src.application",
        ):
            bootstrap_package(package_name)

        _is_bootstrapped = True