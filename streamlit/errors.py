"""Minimal Streamlit errors module for test environment stubs."""


class StreamlitSecretNotFoundError(Exception):
    """Stub exception mirroring Streamlit's secret lookup error."""


__all__ = ["StreamlitSecretNotFoundError"]
