"""Compatibility layer for template loading utilities.

This module preserves the public import path ``automl_platform.template_loader``
expected by the CLI and documentation while delegating the implementation to
``automl_platform.templates.template_loader``.
"""
from .templates.template_loader import TemplateLoader, TemplateConfig, TemplateMetadata

__all__ = ["TemplateLoader", "TemplateConfig", "TemplateMetadata"]
