"""Yamgam - Template rendering tool for docx, xlsx, and pptx files."""

from __future__ import annotations

from yamgam.api import (
    MergeCollision,
    MergeResult,
    ResolutionResult,
    YamgamError,
    YamgamExpressionError,
    YamgamMaxDepthError,
    YamgamSecurityError,
    YamgamYamlError,
    deep_merge,
    extract_expressions,
    extract_referenced_variables,
    merge_layers,
    merge_layers_detailed,
    normalize_dates,
    parse_yaml_layer,
    render_string,
    render_template_text,
    resolve_context,
    resolve_layers,
    validate_expressions,
)
from yamgam.render import (
    DeepMergeLoader,
    NullUndefined,
    YamgamSandboxEnvironment,
    check_jinja_ast_safety,
    raw_yaml,
    recursive_yaml,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DeepMergeLoader",
    "MergeCollision",
    "MergeResult",
    "NullUndefined",
    "ResolutionResult",
    "YamgamError",
    "YamgamExpressionError",
    "YamgamMaxDepthError",
    "YamgamSandboxEnvironment",
    "YamgamSecurityError",
    "YamgamYamlError",
    "check_jinja_ast_safety",
    "deep_merge",
    "extract_expressions",
    "extract_referenced_variables",
    "merge_layers",
    "merge_layers_detailed",
    "normalize_dates",
    "parse_yaml_layer",
    "raw_yaml",
    "recursive_yaml",
    "render_string",
    "render_template_text",
    "resolve_context",
    "resolve_layers",
    "validate_expressions",
]
