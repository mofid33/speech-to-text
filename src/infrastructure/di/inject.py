from __future__ import annotations

import inspect
from functools import wraps
from threading import RLock
from typing import Any, TypeVar, get_type_hints

T = TypeVar("T", bound=type)

_provider_registry: dict[type[Any], type[Any]] = {}
_singleton_instances: dict[type[Any], Any] = {}
_registry_lock = RLock()


def inject(cls: T) -> T:
    original_init = cls.__init__
    signature = inspect.signature(original_init)

    @wraps(original_init)
    def injected_init(self: Any, *args: Any, **kwargs: Any) -> None:
        bound_arguments = signature.bind_partial(self, *args, **kwargs)
        type_hints = get_type_hints(original_init)

        for parameter_name, parameter in signature.parameters.items():
            if parameter_name == "self" or parameter_name in bound_arguments.arguments:
                continue

            if parameter.default is not inspect.Parameter.empty:
                continue

            dependency_type = type_hints.get(parameter_name)
            if dependency_type is None or not isinstance(dependency_type, type):
                raise TypeError(
                    f"Cannot inject parameter '{parameter_name}' for "
                    f"'{cls.__module__}.{cls.__name__}': a concrete type annotation "
                    "is required."
                )

            kwargs[parameter_name] = resolve(dependency_type)

        original_init(self, *args, **kwargs)

    setattr(cls, "__init__", injected_init)
    setattr(cls, "__di_provider__", True)
    return cls


def register_provider(provider_type: type[Any]) -> None:
    with _registry_lock:
        _provider_registry[provider_type] = provider_type

        for base_type in provider_type.__bases__:
            if base_type is object:
                continue

            is_interface = (
                inspect.isabstract(base_type)
                or base_type.__name__.startswith("I")
            )

            if not is_interface:
                continue

            existing_provider = _provider_registry.get(base_type)

            if (
                existing_provider is not None
                and existing_provider is not provider_type
            ):
                raise RuntimeError(
                    f"Multiple providers are registered for interface "
                    f"'{base_type.__module__}.{base_type.__name__}'."
                )

            _provider_registry[base_type] = provider_type


def resolve(dependency_type: type[Any]) -> Any:
    with _registry_lock:
        provider_type = _provider_registry.get(dependency_type)

        if (
            provider_type is None
            and getattr(dependency_type, "__di_provider__", False)
        ):
            provider_type = dependency_type
            _provider_registry[dependency_type] = dependency_type

        if provider_type is None:
            raise LookupError(
                f"No DI provider is registered for "
                f"'{dependency_type.__module__}.{dependency_type.__name__}'."
            )

        if getattr(provider_type, "__di_singleton__", False):
            existing_instance = _singleton_instances.get(provider_type)

            if existing_instance is not None:
                return existing_instance

            instance = provider_type()
            _singleton_instances[provider_type] = instance

            return instance

    return provider_type()


def clear_container() -> None:
    with _registry_lock:
        _provider_registry.clear()
        _singleton_instances.clear()