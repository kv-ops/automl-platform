"""Lightweight Streamlit stub for automated tests.

This module provides a tiny subset of the Streamlit API so that the unit tests
can exercise the dashboard logic without installing the actual Streamlit
package (which is heavy and not required in the execution environment).  The
implementation intentionally focuses on compatibility and only implements the
behaviour that the tests rely on.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence

from .errors import StreamlitSecretNotFoundError

__all__ = [
    "StreamlitSecretNotFoundError",
    "set_page_config",
    "markdown",
    "title",
    "header",
    "subheader",
    "write",
    "info",
    "success",
    "warning",
    "error",
    "divider",
    "metric",
    "progress",
    "selectbox",
    "multiselect",
    "slider",
    "checkbox",
    "button",
    "text_input",
    "file_uploader",
    "dataframe",
    "plotly_chart",
    "download_button",
    "code",
    "image",
    "json",
    "tabs",
    "columns",
    "expander",
    "container",
    "spinner",
    "empty",
    "cache_data",
    "balloons",
    "rerun",
    "session_state",
    "secrets",
    "sidebar",
]


class _SessionState(dict):
    """Dictionary with attribute-style access used by Streamlit."""

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - trivial
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - matches Streamlit behaviour
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:  # pragma: no cover - trivial
        self[key] = value


class _CacheData:
    """Minimal cache stub exposing ``clear`` and decorator semantics."""

    def __call__(self, func=None, **_kwargs):  # pragma: no cover - simple passthrough
        if func is None:

            def decorator(inner):
                return inner

            return decorator
        return func

    def clear(self) -> None:  # pragma: no cover - noop
        return None


class _ContextStub:
    """Context manager used for placeholders, containers and tabs."""

    def __init__(self) -> None:
        self._placeholder = None

    # Context manager protocol -------------------------------------------------
    def __enter__(self) -> "_ContextStub":  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - trivial
        return False

    # Callable behaviour -------------------------------------------------------
    def __call__(self, *args, **kwargs):  # pragma: no cover - noop
        return self

    # Utility methods used in tests -------------------------------------------
    def progress(self, *_args, **_kwargs):  # pragma: no cover - noop
        return None

    def info(self, *_args, **_kwargs):  # pragma: no cover - noop
        return None

    def markdown(self, *_args, **_kwargs):  # pragma: no cover - noop
        return None

    def write(self, *_args, **_kwargs):  # pragma: no cover - noop
        return None

    def success(self, *_args, **_kwargs):  # pragma: no cover - noop
        return None


session_state: _SessionState = _SessionState()
"""Global session state storage used by the dashboard code."""

secrets: dict[str, Any] = {}
"""Dictionary mimicking ``st.secrets``."""

cache_data = _CacheData()


def set_page_config(*_args, **_kwargs) -> None:  # pragma: no cover - noop
    return None


def markdown(*_args, **_kwargs) -> None:  # pragma: no cover - noop
    return None


def title(*_args, **_kwargs) -> None:  # pragma: no cover - noop
    return None


def header(*_args, **_kwargs) -> None:  # pragma: no cover - noop
    return None


def subheader(*_args, **_kwargs) -> None:  # pragma: no cover - noop
    return None


def write(*_args, **_kwargs) -> None:  # pragma: no cover - noop
    return None


def info(*_args, **_kwargs) -> None:  # pragma: no cover - noop
    return None


def success(*_args, **_kwargs) -> None:  # pragma: no cover - noop
    return None


def warning(*_args, **_kwargs) -> None:  # pragma: no cover - noop
    return None


def error(*_args, **_kwargs) -> None:  # pragma: no cover - noop
    return None


def divider() -> None:  # pragma: no cover - noop
    return None


def metric(*_args, **_kwargs) -> None:  # pragma: no cover - noop
    return None


def progress(*_args, **_kwargs) -> None:  # pragma: no cover - noop
    return None


def selectbox(_label: str, options: Sequence[Any], index: int = 0, **_kwargs) -> Any:
    if not options:
        return None
    if 0 <= index < len(options):
        return options[index]
    return options[0]


def multiselect(
    _label: str,
    options: Sequence[Any],
    default: Iterable[Any] | None = None,
    **_kwargs,
) -> List[Any]:
    if default is not None:
        return list(default)
    return list(options) if options else []


def slider(
    _label: str,
    min_value: Any,
    max_value: Any,
    value: Any = None,
    **_kwargs,
) -> Any:
    if value is not None:
        return value
    return min_value


def checkbox(_label: str, value: bool = False, **_kwargs) -> bool:
    return value


def button(_label: str, **_kwargs) -> bool:
    return False


def text_input(_label: str, value: str = "", **_kwargs) -> str:
    return value


def file_uploader(*_args, **_kwargs):  # pragma: no cover - noop
    return None


def dataframe(*_args, **_kwargs) -> None:  # pragma: no cover - noop
    return None


def plotly_chart(*_args, **_kwargs) -> None:  # pragma: no cover - noop
    return None


def download_button(*_args, **_kwargs) -> None:  # pragma: no cover - noop
    return None


def code(*_args, **_kwargs) -> None:  # pragma: no cover - noop
    return None


def image(*_args, **_kwargs) -> None:  # pragma: no cover - noop
    return None


def json(*_args, **_kwargs) -> None:  # pragma: no cover - noop
    return None


def tabs(labels: Sequence[Any]) -> List[_ContextStub]:
    return [_ContextStub() for _ in labels]


def columns(spec: Sequence[Any] | int) -> List[_ContextStub]:
    if isinstance(spec, int):
        count = spec
    else:
        count = len(spec)
    return [_ContextStub() for _ in range(max(count, 0))]


def expander(*_args, **_kwargs) -> _ContextStub:
    return _ContextStub()


def container() -> _ContextStub:
    return _ContextStub()


def spinner(*_args, **_kwargs) -> _ContextStub:
    return _ContextStub()


def empty() -> _ContextStub:
    return _ContextStub()


def balloons() -> None:  # pragma: no cover - noop
    return None


def rerun() -> None:  # pragma: no cover - noop
    return None


sidebar: _ContextStub = _ContextStub()
"""Simple context manager returned when ``with st.sidebar:`` is used."""
