"""Utilities for validating and safely handling LLM generated code."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable, Optional, Set, Tuple, Union


class UnsafeCodeExecutionError(RuntimeError):
    """Raised when a generated code snippet is deemed unsafe to execute."""


@dataclass(frozen=True)
class _ForbiddenPattern:
    name: str
    message: str


_FORBIDDEN_NAMES = {
    "__builtins__",
    "__import__",
    "builtins",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "importlib",
    "input",
    "locals",
    "open",
    "os",
    "pathlib",
    "pickle",
    "setattr",
    "shutil",
    "subprocess",
    "sys",
    "vars",
}

_FORBIDDEN_CALLS: Iterable[_ForbiddenPattern] = (
    _ForbiddenPattern("builtins.__import__", "Dynamic imports are not permitted."),
    _ForbiddenPattern("os.popen", "System commands are not permitted."),
    _ForbiddenPattern("os.system", "System commands are not permitted."),
    _ForbiddenPattern("pickle.load", "Deserialising data is not permitted."),
    _ForbiddenPattern("pickle.loads", "Deserialising data is not permitted."),
    _ForbiddenPattern("subprocess.Popen", "Subprocess execution is not permitted."),
    _ForbiddenPattern("subprocess.call", "Subprocess execution is not permitted."),
    _ForbiddenPattern("subprocess.check_output", "Subprocess execution is not permitted."),
    _ForbiddenPattern("subprocess.run", "Subprocess execution is not permitted."),
)

_FORBIDDEN_STRINGS = {
    "__builtins__",
    "__import__",
    "eval",
    "exec",
    "pickle",
}

_ALLOWED_NODE_TYPES: Tuple[type, ...] = (
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Return,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.If,
    ast.IfExp,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.TryStar,
    ast.With,
    ast.AsyncWith,
    ast.withitem,
    ast.Raise,
    ast.Call,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.ExtSlice,
    ast.Tuple,
    ast.List,
    ast.Set,
    ast.Dict,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.Name,
    ast.NamedExpr,
    ast.Load,
    ast.Store,
    ast.Del,
    ast.Delete,
    ast.Compare,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.MatMult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.LShift,
    ast.RShift,
    ast.BitOr,
    ast.BitXor,
    ast.BitAnd,
    ast.Invert,
    ast.USub,
    ast.UAdd,
    ast.Eq,
    ast.NotEq,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
    ast.Constant,
    ast.JoinedStr,
    ast.FormattedValue,
    ast.Starred,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Assert,
    ast.ExceptHandler,
    ast.Import,
    ast.ImportFrom,
    ast.alias,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.arguments,
    ast.arg,
    ast.keyword,
    ast.Lambda,
    ast.Match,
    ast.match_case,
    ast.MatchValue,
    ast.MatchAs,
    ast.MatchOr,
    ast.MatchSingleton,
    ast.MatchSequence,
    ast.MatchMapping,
    ast.MatchStar,
    ast.MatchClass,
)

_UNSUPPORTED_NODES: Set[type] = {
    ast.ClassDef,
    ast.Global,
    ast.Nonlocal,
}


def _get_full_name(node: ast.AST) -> Optional[str]:
    """Return the dotted path for the given AST node when possible."""

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _get_full_name(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    return None


def _ensure_supported_nodes(tree: ast.AST) -> None:
    """Ensure only nodes in the explicit allow-list appear in the tree."""

    for node in ast.walk(tree):
        node_type = type(node)
        if node_type in _UNSUPPORTED_NODES or node_type not in _ALLOWED_NODE_TYPES:
            raise UnsafeCodeExecutionError(
                f"Use of unsupported Python construct '{node_type.__name__}' detected."
            )


def _check_forbidden_names(name: str, message: str) -> None:
    if name in _FORBIDDEN_NAMES:
        raise UnsafeCodeExecutionError(message)


def _check_constant(value: Union[str, bytes, int, float, complex, None, bool]) -> None:
    if isinstance(value, str):
        lowered = value.lower()
        for forbidden in _FORBIDDEN_STRINGS:
            if forbidden in lowered:
                raise UnsafeCodeExecutionError(
                    "Suspicious string literal detected in generated code."
                )


def ensure_safe_cleaning_code(code: str) -> None:
    """Validate that LLM generated cleaning code does not perform unsafe actions.

    The function parses the snippet and rejects imports, filesystem/process
    interactions and other high-risk primitives.  If the code is deemed unsafe
    an :class:`UnsafeCodeExecutionError` is raised with a human readable message.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:  # pragma: no cover - handled in calling code
        raise UnsafeCodeExecutionError(
            f"Generated code contains syntax errors: {exc}"
        ) from exc

    _ensure_supported_nodes(tree)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeExecutionError("Imports are not allowed in generated cleaning code.")

        if isinstance(node, ast.Call):
            func_name = _get_full_name(node.func)
            if func_name:
                for pattern in _FORBIDDEN_CALLS:
                    if func_name == pattern.name:
                        raise UnsafeCodeExecutionError(pattern.message)
                root = func_name.split(".")[0]
                _check_forbidden_names(
                    root,
                    f"Call to forbidden function '{func_name}' detected.",
                )
            elif isinstance(node.func, ast.Lambda):
                raise UnsafeCodeExecutionError("Lambda expressions are not allowed in generated code.")

        if isinstance(node, ast.Attribute):
            attribute_name = _get_full_name(node)
            if attribute_name:
                root, *_ = attribute_name.split(".")
                _check_forbidden_names(
                    root,
                    f"Access to forbidden module '{root}' is not permitted.",
                )
                if node.attr.startswith("__"):
                    raise UnsafeCodeExecutionError(
                        "Access to dunder attributes is not permitted in generated code."
                    )

        if isinstance(node, ast.Name):
            _check_forbidden_names(
                node.id,
                f"Reference to forbidden name '{node.id}' detected.",
            )

        if isinstance(node, ast.Subscript):
            target_name = _get_full_name(node.value)
            if target_name:
                root, *_ = target_name.split(".")
                _check_forbidden_names(
                    root,
                    "Subscript access on forbidden object is not permitted.",
                )

        if isinstance(node, ast.Constant):
            _check_constant(node.value)


def execution_not_allowed_message() -> str:
    """Return the standard message used when execution is disabled."""

    return (
        "Automatic execution of generated cleaning code is disabled for safety. "
        "Please review and apply the transformations manually."
    )
