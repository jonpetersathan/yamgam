"""Clean in-memory Python API for yamgam YAML merging, recursive Jinja2 evaluation, and validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass, field
from datetime import date, datetime
import re
from typing import Any

from jinja2 import ChainableUndefined, StrictUndefined, TemplateSyntaxError, Undefined, meta, nodes
from jinja2.environment import TemplateExpression
from jinja2.exceptions import SecurityError, UndefinedError
import yaml

from yamgam.render import (
    BLOCKED_ATTR_NAMES,
    BLOCKED_CALLABLE_NAMES,
    DeepMergeLoader,
    YamgamSandboxEnvironment,
    check_jinja_ast_safety,
)

_JINJA_MARKER_RE = re.compile(r"(\{\{|\{%|\{#)")


class YamgamError(Exception):
    """Base exception for all yamgam library errors."""


class YamgamYamlError(YamgamError):
    """Raised when YAML parsing or root structure validation fails."""

    def __init__(self, message: str, *, line: int = 1, column: int = 1) -> None:
        super().__init__(message)
        self.line = line
        self.column = column


class YamgamSecurityError(YamgamError, SecurityError):
    """Raised when a template violates sandbox security rules (e.g., SSTI)."""


class YamgamExpressionError(YamgamError):
    """Raised when Jinja2 template syntax or evaluation fails."""

    def __init__(self, message: str, *, path: str | None = None) -> None:
        super().__init__(message)
        self.path = path


class YamgamMaxDepthError(YamgamError):
    """Raised when recursive evaluation hits a cycle or exceeds max_depth."""

    def __init__(
        self,
        message: str,
        *,
        depth: int | None = None,
        unresolved_paths: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.depth = depth
        self.unresolved_paths = unresolved_paths or []


@dataclass(frozen=True, slots=True)
class MergeCollision:
    """Describes a key overwrite between two ordered layers."""

    path: str
    overridden_index: int
    winning_index: int
    previous_value: Any
    new_value: Any


@dataclass(frozen=True, slots=True)
class MergeResult:
    """Result of priority deep-merging multiple ordered layers."""

    merged: dict[str, Any]
    collisions: list[MergeCollision] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """Result of dependency-aware recursive context evaluation."""

    resolved: dict[str, Any]
    raw: dict[str, Any]
    depth_used: int
    converged: bool
    unresolved_paths: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    collisions: list[MergeCollision] = field(default_factory=list)


class _TrackingChainableUndefined(ChainableUndefined):
    """Non-strict ChainableUndefined that records missing variable paths without leaking sentinel tokens."""

    __slots__ = ("_tracker",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracker: list[str] | None = None

    def _record_missing(self) -> None:
        if self._tracker is not None and self._undefined_name:
            name = str(self._undefined_name)
            if name not in self._tracker:
                self._tracker.append(name)

    def _fail_with_undefined_error(self, *args: Any, **kwargs: Any) -> Any:
        self._record_missing()
        return super()._fail_with_undefined_error(*args, **kwargs)

    def __str__(self) -> str:
        self._record_missing()
        return ""

    def __iter__(self):
        self._record_missing()
        return iter(())

    def __bool__(self) -> bool:
        self._record_missing()
        return False


def normalize_dates(value: Any) -> Any:
    """Recursively normalize datetime.datetime and datetime.date to 'YYYY-MM-DD' ISO strings."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {k: normalize_dates(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_dates(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_dates(item) for item in value)
    return copy.deepcopy(value)


def _normalize_dict_keys(value: Any) -> Any:
    """Recursively convert Mapping keys to str and copy mutable containers."""
    if isinstance(value, Mapping):
        return {str(k): _normalize_dict_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_dict_keys(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_dict_keys(item) for item in value]
    return copy.deepcopy(value)


def parse_yaml_layer(
    raw_yaml_text: str,
    *,
    normalize_date_values: bool = True,
) -> dict[str, Any]:
    """Safely parse a YAML string layer into a JSON-safe dict without file I/O."""
    if not isinstance(raw_yaml_text, str):
        raise YamgamYamlError("YAML layer must be a string.")
    try:
        loaded = yaml.load(raw_yaml_text, Loader=DeepMergeLoader)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = int(mark.line) + 1 if mark is not None else 1
        column = int(mark.column) + 1 if mark is not None else 1
        raise YamgamYamlError(str(exc), line=line, column=column) from exc
    except TypeError as exc:
        raise YamgamYamlError(f"Invalid YAML mapping or unhashable key: {exc}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise YamgamYamlError("YAML root must be a mapping/dictionary.")

    normalized = _normalize_dict_keys(loaded)
    if normalize_date_values:
        normalized = normalize_dates(normalized)
    return normalized


def _deep_merge_into(
    target: dict[str, Any],
    override: Mapping[str, Any],
    *,
    prefix: str,
    override_index: int,
    last_writer: dict[str, int],
    collisions: list[MergeCollision],
) -> None:
    for raw_key, raw_val in override.items():
        key = str(raw_key)
        full_path = f"{prefix}.{key}" if prefix else key
        val = _normalize_dict_keys(raw_val)

        if key in target:
            existing = target[key]
            if isinstance(existing, dict) and isinstance(val, Mapping):
                _deep_merge_into(
                    existing,
                    dict(val),
                    prefix=full_path,
                    override_index=override_index,
                    last_writer=last_writer,
                    collisions=collisions,
                )
                continue

            prev_index = last_writer.get(full_path, 0)
            collisions.append(
                MergeCollision(
                    path=full_path,
                    overridden_index=prev_index,
                    winning_index=override_index,
                    previous_value=copy.deepcopy(existing),
                    new_value=copy.deepcopy(val),
                )
            )
            target[key] = dict(val) if isinstance(val, Mapping) else val
            _set_subtree_writer(full_path, target[key], override_index, last_writer)
        else:
            target[key] = dict(val) if isinstance(val, Mapping) else val
            _set_subtree_writer(full_path, target[key], override_index, last_writer)


def _set_subtree_writer(
    prefix: str,
    value: Any,
    layer_index: int,
    last_writer: dict[str, int],
) -> None:
    if prefix:
        last_writer[prefix] = layer_index
    if isinstance(value, dict):
        for k, v in value.items():
            child_path = f"{prefix}.{k}" if prefix else str(k)
            _set_subtree_writer(child_path, v, layer_index, last_writer)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Non-mutating deep merge of `override` onto `base` (dicts merge, lists/scalars replace)."""
    if not isinstance(base, Mapping) or not isinstance(override, Mapping):
        raise TypeError("deep_merge requires Mapping arguments.")
    merged = _normalize_dict_keys(base)
    last_writer: dict[str, int] = {}
    _set_subtree_writer("", merged, 0, last_writer)
    collisions: list[MergeCollision] = []
    _deep_merge_into(
        merged,
        override,
        prefix="",
        override_index=1,
        last_writer=last_writer,
        collisions=collisions,
    )
    return merged


def _coerce_layer(
    layer: Mapping[str, Any] | str,
    *,
    normalize_date_values: bool,
) -> dict[str, Any]:
    if isinstance(layer, str):
        return parse_yaml_layer(layer, normalize_date_values=normalize_date_values)
    if isinstance(layer, Mapping):
        normalized = _normalize_dict_keys(layer)
        return normalize_dates(normalized) if normalize_date_values else normalized
    raise TypeError(f"Layer must be a Mapping or YAML str, got {type(layer).__name__}.")


def merge_layers_detailed(
    layers: Sequence[Mapping[str, Any] | str],
    *,
    highest_first: bool = False,
    normalize_date_values: bool = True,
) -> MergeResult:
    """Deep-merge ordered layers without mutating inputs, returning merged dict and collision metadata."""
    indexed = list(enumerate(layers))
    ordered = list(reversed(indexed)) if highest_first else indexed

    merged: dict[str, Any] = {}
    last_writer: dict[str, int] = {}
    collisions: list[MergeCollision] = []

    for orig_idx, raw_layer in ordered:
        layer_dict = _coerce_layer(raw_layer, normalize_date_values=normalize_date_values)
        _deep_merge_into(
            merged,
            layer_dict,
            prefix="",
            override_index=orig_idx,
            last_writer=last_writer,
            collisions=collisions,
        )

    return MergeResult(merged=merged, collisions=collisions)


def merge_layers(
    layers: Sequence[Mapping[str, Any] | str],
    *,
    highest_first: bool = False,
    normalize_date_values: bool = True,
    return_collisions: bool = False,
) -> dict[str, Any] | MergeResult:
    """Deep-merge ordered dict/YAML layers without mutating caller inputs."""
    result = merge_layers_detailed(
        layers,
        highest_first=highest_first,
        normalize_date_values=normalize_date_values,
    )
    if return_collisions:
        return result
    return result.merged


def _contains_jinja(value: str) -> bool:
    return bool(_JINJA_MARKER_RE.search(value))


def _subtree_contains_jinja(value: Any) -> bool:
    if isinstance(value, str):
        return _contains_jinja(value)
    if isinstance(value, Mapping):
        return any(_subtree_contains_jinja(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_subtree_contains_jinja(item) for item in value)
    return False


def _extract_expressions_from_string(
    text: str,
    sandbox_env: YamgamSandboxEnvironment,
) -> list[str]:
    if not _contains_jinja(text):
        return []
    found: list[str] = []
    try:
        tokens = list(sandbox_env.lexer.tokeniter(text, name=None))
    except TemplateSyntaxError as exc:
        raise YamgamExpressionError(str(exc)) from exc

    i = 0
    while i < len(tokens):
        _, ttype, val = tokens[i]
        if ttype in ("variable_begin", "block_begin", "comment_begin"):
            end_type = {
                "variable_begin": "variable_end",
                "block_begin": "block_end",
                "comment_begin": "comment_end",
            }[ttype]
            parts = [val]
            i += 1
            found_end = False
            while i < len(tokens):
                _, inner_type, inner_val = tokens[i]
                parts.append(inner_val)
                if inner_type == end_type:
                    found_end = True
                    break
                i += 1
            if not found_end:
                raise YamgamExpressionError(f"Unclosed Jinja2 tag starting with {val!r}.")
            if ttype != "comment_begin":
                found.append("".join(parts))
        i += 1
    return found


def extract_expressions(
    value: Any,
    *,
    env: YamgamSandboxEnvironment | None = None,
) -> list[str]:
    """Walk an in-memory value and return raw Jinja2 expressions/tags found."""
    sandbox_env = env or YamgamSandboxEnvironment()
    results: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, str):
            results.extend(_extract_expressions_from_string(node, sandbox_env))
        elif isinstance(node, Mapping):
            for v in node.values():
                _walk(v)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _walk(item)

    _walk(value)
    return results


def _extract_target_names(target: nodes.Node) -> set[str]:
    names: set[str] = set()
    if isinstance(target, nodes.Name):
        names.add(target.name)
    elif isinstance(target, nodes.Tuple):
        for item in target.items:
            names.update(_extract_target_names(item))
    return names


_SIMPLE_IDENT_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _format_segment_path(segments: tuple[str | int, ...]) -> str:
    """Format structured path segments into a readable dotted/indexed path string."""
    if not segments:
        return ""
    parts: list[str] = [str(segments[0])]
    for seg in segments[1:]:
        if isinstance(seg, int):
            parts.append(f"[{seg}]")
        elif _SIMPLE_IDENT_RE.match(seg):
            parts.append(f".{seg}")
        else:
            parts.append(f"[{seg!r}]")
    return "".join(parts)


def _extract_reference_segments_from_ast(
    parsed_ast: nodes.Template,
    ignored_roots: set[str],
) -> list[tuple[str | int, ...]]:
    """Extract ordered, deduplicated structured variable paths respecting lexical For/Assign scope."""
    found: list[tuple[str | int, ...]] = []
    seen: set[tuple[str | int, ...]] = set()

    def _record(segments: tuple[str | int, ...], active_ignored: set[str]) -> None:
        root = segments[0]
        if not isinstance(root, str) or root in active_ignored or root.startswith("_"):
            return
        if segments not in seen:
            seen.add(segments)
            found.append(segments)

    def _try_chain_segments(node: nodes.Node) -> tuple[str | int, ...] | None:
        if isinstance(node, nodes.Name):
            if node.ctx == "load":
                return (node.name,)
            return None
        if isinstance(node, nodes.Getattr):
            prefix = _try_chain_segments(node.node)
            if prefix is not None:
                return prefix + (node.attr,)
        if isinstance(node, nodes.Getitem):
            prefix = _try_chain_segments(node.node)
            if prefix is not None and isinstance(node.arg, nodes.Const):
                if isinstance(node.arg.value, int) and not isinstance(node.arg.value, bool):
                    return prefix + (node.arg.value,)
                if isinstance(node.arg.value, str):
                    return prefix + (node.arg.value,)
        return None

    def _visit_stmt_list(stmts: Sequence[nodes.Node], scope_ignored: set[str]) -> None:
        block_ignored = set(scope_ignored)
        for stmt in stmts:
            if isinstance(stmt, nodes.Assign):
                _visit_node(stmt.node, block_ignored)
                block_ignored.update(_extract_target_names(stmt.target))
            elif isinstance(stmt, nodes.AssignBlock):
                if stmt.filter is not None:
                    _visit_node(stmt.filter, block_ignored)
                _visit_stmt_list(stmt.body, block_ignored)
                block_ignored.update(_extract_target_names(stmt.target))
            else:
                _visit_node(stmt, block_ignored)

    def _visit_node(node: nodes.Node, active_ignored: set[str]) -> None:
        if isinstance(node, nodes.Template):
            _visit_stmt_list(node.body, active_ignored)
            return
        if isinstance(node, nodes.Scope):
            _visit_stmt_list(node.body, active_ignored)
            return
        if isinstance(node, nodes.For):
            _visit_node(node.iter, active_ignored)
            loop_ignored = active_ignored | _extract_target_names(node.target) | {"loop"}
            if node.test is not None:
                _visit_node(node.test, loop_ignored)
            _visit_stmt_list(node.body, loop_ignored)
            if node.else_:
                _visit_stmt_list(node.else_, active_ignored)
            return
        if isinstance(node, nodes.If):
            _visit_node(node.test, active_ignored)
            _visit_stmt_list(node.body, active_ignored)
            for elif_node in node.elif_:
                _visit_node(elif_node, active_ignored)
            if node.else_:
                _visit_stmt_list(node.else_, active_ignored)
            return
        if isinstance(node, (nodes.Name, nodes.Getattr, nodes.Getitem)):
            chain = _try_chain_segments(node)
            if chain is not None:
                _record(chain, active_ignored)
                return
        for child in node.iter_child_nodes():
            _visit_node(child, active_ignored)

    _visit_node(parsed_ast, set(ignored_roots))

    try:
        undeclared = meta.find_undeclared_variables(parsed_ast)
    except (ValueError, ArithmeticError, OverflowError) as exc:
        raise YamgamSecurityError(f"Unsafe expression in template: {exc}") from exc

    for var in sorted(undeclared):
        if (
            var not in ignored_roots
            and not var.startswith("_")
            and not any(segs[0] == var for segs in found)
        ):
            _record((var,), ignored_roots)

    return found


def _extract_dotted_references_from_ast(
    parsed_ast: nodes.Template,
    ignored_roots: set[str],
) -> list[str]:
    """Extract ordered, deduplicated dotted variable paths referenced in a Jinja2 AST."""
    return [
        _format_segment_path(segs)
        for segs in _extract_reference_segments_from_ast(parsed_ast, ignored_roots)
    ]


def _parse_and_check(
    template_str: str,
    sandbox_env: YamgamSandboxEnvironment,
    *,
    allow_statements: bool = True,
    path: str | None = None,
) -> nodes.Template:
    try:
        parsed = sandbox_env.parse(template_str)
    except SecurityError as exc:
        raise YamgamSecurityError(str(exc)) from exc
    except TemplateSyntaxError as exc:
        raise YamgamExpressionError(str(exc), path=path) from exc

    try:
        check_jinja_ast_safety(parsed)
    except SecurityError as exc:
        raise YamgamSecurityError(str(exc)) from exc

    if not allow_statements:
        for stmt in parsed.body:
            if not isinstance(stmt, nodes.Output):
                raise YamgamSecurityError(
                    f"Jinja2 statement '{type(stmt).__name__}' is disallowed when allow_statements=False."
                )
    return parsed


def extract_referenced_variables(
    value: Any,
    *,
    env: YamgamSandboxEnvironment | None = None,
) -> list[str]:
    """Extract ordered, deduplicated variable paths referenced in a template string or dict/list."""
    sandbox_env = env or YamgamSandboxEnvironment()
    context_roots = {str(k) for k in value.keys()} if isinstance(value, Mapping) else set()
    ignored_roots = (
        set(sandbox_env.globals.keys()) | {"loop", "true", "false", "none"}
    ) - context_roots
    found: list[str] = []
    seen: set[str] = set()

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, str):
            if not _contains_jinja(node):
                return
            parsed = _parse_and_check(node, sandbox_env, allow_statements=True, path=path or None)
            for ref in _extract_dotted_references_from_ast(parsed, ignored_roots):
                if ref not in seen:
                    seen.add(ref)
                    found.append(ref)
        elif isinstance(node, Mapping):
            for k, v in node.items():
                child_path = f"{path}.{k}" if path else str(k)
                _walk(v, child_path)
        elif isinstance(node, (list, tuple)):
            for idx, item in enumerate(node):
                child_path = f"{path}[{idx}]"
                _walk(item, child_path)

    _walk(value, "")
    return found


def validate_expressions(
    value: Any,
    *,
    allow_statements: bool = True,
    raise_on_error: bool = True,
    env: YamgamSandboxEnvironment | None = None,
) -> list[dict[str, str]]:
    """Statically validate all Jinja2 templates in memory (rejects SSTI even in unexecuted branches)."""
    sandbox_env = env or YamgamSandboxEnvironment()
    errors: list[dict[str, str]] = []

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, str):
            if not _contains_jinja(node):
                return
            try:
                _parse_and_check(
                    node,
                    sandbox_env,
                    allow_statements=allow_statements,
                    path=path or None,
                )
            except YamgamSecurityError as exc:
                if raise_on_error:
                    raise
                errors.append(
                    {
                        "path": path or "<root>",
                        "error_type": "security",
                        "message": str(exc),
                    }
                )
            except YamgamExpressionError as exc:
                if raise_on_error:
                    raise
                errors.append(
                    {
                        "path": path or "<root>",
                        "error_type": "syntax",
                        "message": str(exc),
                    }
                )
        elif isinstance(node, Mapping):
            for k, v in node.items():
                child_path = f"{path}.{k}" if path else str(k)
                _walk(v, child_path)
        elif isinstance(node, (list, tuple)):
            for idx, item in enumerate(node):
                child_path = f"{path}[{idx}]"
                _walk(item, child_path)

    _walk(value, "")
    return errors


def _build_env(
    *,
    strict_undefined: bool,
    extra_globals: Mapping[str, Any] | None = None,
) -> YamgamSandboxEnvironment:
    undefined_cls = StrictUndefined if strict_undefined else _TrackingChainableUndefined
    env = YamgamSandboxEnvironment(undefined=undefined_cls)
    if extra_globals:
        for k, v in extra_globals.items():
            if not isinstance(k, str) or k.startswith("_") or k in BLOCKED_CALLABLE_NAMES:
                raise YamgamSecurityError(f"Global name '{k}' is forbidden in yamgam.")
            env.globals[k] = v
    return env


def _is_pure_expression(text: str, env: YamgamSandboxEnvironment) -> tuple[bool, str]:
    stripped = text.strip()
    if not (stripped.startswith("{{") and stripped.endswith("}}")):
        return False, ""
    parsed = env.parse(text)
    is_pure = (
        len(parsed.body) == 1
        and isinstance(parsed.body[0], nodes.Output)
        and len(parsed.body[0].nodes) == 1
        and not isinstance(parsed.body[0].nodes[0], nodes.TemplateData)
    )
    if not is_pure:
        return False, ""
    tokens = list(env.lexer.tokeniter(stripped, name=None))
    if (
        len(tokens) < 2
        or tokens[0][1] != "variable_begin"
        or tokens[-1][1] != "variable_end"
    ):
        return False, ""
    inner = "".join(val for _, _, val in tokens[1:-1]).strip()
    return True, inner


def _is_path_evaluated(
    path_segs: tuple[str | int, ...],
    evaluated_paths: set[tuple[str | int, ...]] | frozenset[tuple[str | int, ...]],
) -> bool:
    for idx in range(1, len(path_segs) + 1):
        if path_segs[:idx] in evaluated_paths:
            return True
    return False


def _subtree_has_unresolved_jinja(
    value: Any,
    path_segs: tuple[str | int, ...],
    evaluated_paths: set[tuple[str | int, ...]] | frozenset[tuple[str | int, ...]],
) -> bool:
    if path_segs and _is_path_evaluated(path_segs, evaluated_paths):
        return False
    if isinstance(value, str):
        return _contains_jinja(value)
    if isinstance(value, Mapping):
        return any(
            _subtree_has_unresolved_jinja(v, path_segs + (str(k),), evaluated_paths)
            for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(
            _subtree_has_unresolved_jinja(item, path_segs + (idx,), evaluated_paths)
            for idx, item in enumerate(value)
        )
    return False


def _ref_segments_has_unresolved_jinja(
    snapshot: Mapping[str, Any],
    segments: tuple[str | int, ...],
    evaluated_paths: set[tuple[str | int, ...]] | frozenset[tuple[str | int, ...]],
) -> bool:
    """Return True if `segments` (or an ancestor/descendant along it) still holds unresolved Jinja2."""
    current: Any = snapshot
    traversed: list[str | int] = []
    for seg in segments:
        traversed_tup = tuple(traversed)
        if traversed_tup and _is_path_evaluated(traversed_tup, evaluated_paths):
            return False
        if isinstance(current, str) and _contains_jinja(current):
            return True
        if isinstance(seg, int):
            if isinstance(current, (list, tuple)) and -len(current) <= seg < len(current):
                norm_idx = seg if seg >= 0 else len(current) + seg
                current = current[norm_idx]
                traversed.append(norm_idx)
            else:
                return False
        else:
            if isinstance(current, Mapping) and seg in current:
                current = current[seg]
                traversed.append(seg)
            else:
                return False
    return _subtree_has_unresolved_jinja(current, tuple(traversed), evaluated_paths)


def _find_unresolved_paths(
    value: Any,
    path: str = "",
    *,
    path_segs: tuple[str | int, ...] = (),
    evaluated_paths: set[tuple[str | int, ...]] | frozenset[tuple[str | int, ...]] = frozenset(),
) -> list[str]:
    if path_segs and _is_path_evaluated(path_segs, evaluated_paths):
        return []
    paths: list[str] = []
    if isinstance(value, str):
        if _contains_jinja(value):
            paths.append(path or "<root>")
    elif isinstance(value, Mapping):
        for k, v in value.items():
            key_str = str(k)
            child_path = f"{path}.{key_str}" if path else key_str
            paths.extend(
                _find_unresolved_paths(
                    v,
                    child_path,
                    path_segs=path_segs + (key_str,),
                    evaluated_paths=evaluated_paths,
                )
            )
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            paths.extend(
                _find_unresolved_paths(
                    item,
                    f"{path}[{idx}]",
                    path_segs=path_segs + (idx,),
                    evaluated_paths=evaluated_paths,
                )
            )
    return paths


def _prepare_render_ctx(
    context: Mapping[str, Any],
    env: YamgamSandboxEnvironment,
) -> tuple[Mapping[str, Any], bool]:
    has_self_override = "self" in context or "self" in env.globals
    if not has_self_override:
        return context, False
    render_ctx = dict(context)
    render_ctx["__yamgam_self__"] = (
        context["self"] if "self" in context else env.globals["self"]
    )
    return render_ctx, True


def _compile_pure_expression(
    expr_src: str,
    env: YamgamSandboxEnvironment,
    *,
    has_self_override: bool,
) -> TemplateExpression:
    if not has_self_override:
        return env.compile_expression(expr_src, undefined_to_none=False)
    parsed_expr = env.parse(f"{{{{ {expr_src} }}}}")
    if (
        len(parsed_expr.body) != 1
        or not isinstance(parsed_expr.body[0], nodes.Output)
        or len(parsed_expr.body[0].nodes) != 1
    ):
        return env.compile_expression(expr_src, undefined_to_none=False)
    expr_node = parsed_expr.body[0].nodes[0]
    body = [
        nodes.Assign(
            nodes.Name("self", "store"),
            nodes.Name("__yamgam_self__", "load"),
            lineno=1,
        ),
        nodes.Assign(nodes.Name("result", "store"), expr_node, lineno=1),
    ]
    template = env.from_string(nodes.Template(body, lineno=1))
    return TemplateExpression(template, False)


def _compile_template(
    template_str: str,
    env: YamgamSandboxEnvironment,
    *,
    has_self_override: bool,
) -> Any:
    if not has_self_override:
        return env.from_string(template_str)
    parsed = env.parse(template_str)
    parsed.body.insert(
        0,
        nodes.Assign(
            nodes.Name("self", "store"),
            nodes.Name("__yamgam_self__", "load"),
            lineno=1,
        ),
    )
    return env.from_string(parsed)


def _eval_single_template(
    template_str: str,
    context: Mapping[str, Any],
    *,
    env: YamgamSandboxEnvironment,
    strict_undefined: bool,
    normalize_date_values: bool,
    path: str | None = None,
) -> tuple[Any, list[str], str | None]:
    missing_refs: list[str] = []
    runtime_error: str | None = None

    def _run_eval() -> Any:
        nonlocal runtime_error
        try:
            render_ctx, has_self_override = _prepare_render_ctx(context, env)
            is_pure, expr_src = _is_pure_expression(template_str, env)
            if is_pure:
                compiled = _compile_pure_expression(
                    expr_src, env, has_self_override=has_self_override
                )
                val = compiled(render_ctx)
                if isinstance(val, Undefined):
                    if strict_undefined:
                        raise UndefinedError(val._undefined_message)
                    missing_refs.append(str(val._undefined_name or expr_src))
                    return None
                return val
            template = _compile_template(
                template_str, env, has_self_override=has_self_override
            )
            return template.render(render_ctx)
        except SecurityError as exc:
            raise YamgamSecurityError(str(exc)) from exc
        except UndefinedError as exc:
            if strict_undefined:
                raise YamgamExpressionError(
                    f"Undefined variable in '{path or '<template>'}': {exc}",
                    path=path,
                ) from exc
            missing_refs.append(str(exc))
            return None
        except TemplateSyntaxError as exc:
            raise YamgamExpressionError(
                f"Template syntax error in '{path or '<template>'}': {exc}",
                path=path,
            ) from exc
        except (ArithmeticError, TypeError, ValueError, LookupError) as exc:
            if strict_undefined:
                raise YamgamExpressionError(
                    f"Expression evaluation error in '{path or '<template>'}': {exc}",
                    path=path,
                ) from exc
            runtime_error = f"{type(exc).__name__}: {exc}"
            return None

    if not strict_undefined:
        tracker: list[str] = []

        class _BoundTrackingUndefined(_TrackingChainableUndefined):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._tracker = tracker

        prev_undef = env.undefined
        env.undefined = _BoundTrackingUndefined
        try:
            result = _run_eval()
        finally:
            env.undefined = prev_undef
        for item in tracker:
            if item not in missing_refs:
                missing_refs.append(item)
    else:
        result = _run_eval()

    if normalize_date_values:
        result = normalize_dates(result)
    return result, missing_refs, runtime_error


def resolve_context(
    layers: Sequence[Mapping[str, Any] | str] | Mapping[str, Any],
    *,
    max_depth: int = 10,
    highest_first: bool = False,
    strict_undefined: bool = True,
    raise_on_max_depth: bool = True,
    normalize_date_values: bool = True,
    extra_globals: Mapping[str, Any] | None = None,
) -> ResolutionResult:
    """Merge ordered layers and recursively evaluate Jinja2 templates in memory using SandboxedEnvironment."""
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1.")

    layer_seq: Sequence[Mapping[str, Any] | str]
    if isinstance(layers, Mapping):
        layer_seq = [layers]
    else:
        layer_seq = layers

    merge_result = merge_layers_detailed(
        layer_seq,
        highest_first=highest_first,
        normalize_date_values=normalize_date_values,
    )
    raw = copy.deepcopy(merge_result.merged)
    current = copy.deepcopy(merge_result.merged)

    env = _build_env(strict_undefined=strict_undefined, extra_globals=extra_globals)
    ignored_roots = (
        set(env.globals.keys()) | {"loop", "true", "false", "none"}
    ) - set(current.keys())

    # Static AST security + syntax validation before executing any pass
    validate_expressions(current, allow_statements=True, raise_on_error=True, env=env)

    errors: dict[str, str] = {}
    error_paths: list[str] = []
    evaluated_paths: set[tuple[str | int, ...]] = set()
    depth_used = 0

    if _subtree_has_unresolved_jinja(current, (), evaluated_paths):
        for pass_idx in range(1, max_depth + 1):
            depth_used = pass_idx
            snapshot = copy.deepcopy(current)
            prev_evaluated_paths = frozenset(evaluated_paths)
            made_progress = False

            def _resolve_node(
                node: Any,
                path: str,
                path_segs: tuple[str | int, ...],
            ) -> Any:
                nonlocal made_progress
                if path_segs and _is_path_evaluated(path_segs, prev_evaluated_paths):
                    return node
                if isinstance(node, str):
                    if not _contains_jinja(node):
                        return node
                    parsed = _parse_and_check(node, env, allow_statements=True, path=path)
                    ref_segs = _extract_reference_segments_from_ast(parsed, ignored_roots)
                    if any(
                        _ref_segments_has_unresolved_jinja(snapshot, segs, prev_evaluated_paths)
                        for segs in ref_segs
                    ):
                        return node
                    evaluated, missing, runtime_error = _eval_single_template(
                        node,
                        snapshot,
                        env=env,
                        strict_undefined=strict_undefined,
                        normalize_date_values=normalize_date_values,
                        path=path,
                    )
                    evaluated_paths.add(path_segs)
                    if missing:
                        errors[path] = f"Undefined variable(s): {', '.join(missing)}"
                        if path not in error_paths:
                            error_paths.append(path)
                    elif runtime_error:
                        errors[path] = runtime_error
                        if path not in error_paths:
                            error_paths.append(path)
                    if evaluated != node or type(evaluated) is not type(node) or path_segs not in prev_evaluated_paths:
                        made_progress = True
                    return evaluated
                if isinstance(node, dict):
                    return {
                        k: _resolve_node(
                            v,
                            f"{path}.{k}" if path else k,
                            path_segs + (str(k),),
                        )
                        for k, v in node.items()
                    }
                if isinstance(node, list):
                    return [
                        _resolve_node(
                            item,
                            f"{path}[{idx}]",
                            path_segs + (idx,),
                        )
                        for idx, item in enumerate(node)
                    ]
                return node

            current = _resolve_node(current, "", ())
            if not _subtree_has_unresolved_jinja(current, (), evaluated_paths):
                break
            if not made_progress:
                break

    remaining_unresolved = _find_unresolved_paths(current, evaluated_paths=evaluated_paths)
    unresolved_paths = list(dict.fromkeys(remaining_unresolved + error_paths))
    converged = not remaining_unresolved and not error_paths

    if remaining_unresolved and raise_on_max_depth:
        raise YamgamMaxDepthError(
            f"Recursive evaluation did not converge within max_depth={max_depth}: {remaining_unresolved}",
            depth=depth_used,
            unresolved_paths=remaining_unresolved,
        )

    if normalize_date_values:
        current = normalize_dates(current)

    return ResolutionResult(
        resolved=current,
        raw=raw,
        depth_used=depth_used,
        converged=converged,
        unresolved_paths=unresolved_paths,
        errors=errors,
        collisions=merge_result.collisions,
    )


def resolve_layers(
    layers: Sequence[Mapping[str, Any] | str] | Mapping[str, Any],
    *,
    max_depth: int = 10,
    highest_first: bool = False,
    strict_undefined: bool = True,
    raise_on_max_depth: bool = True,
    normalize_date_values: bool = True,
    extra_globals: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper returning only the resolved dict from `resolve_context`."""
    return resolve_context(
        layers,
        max_depth=max_depth,
        highest_first=highest_first,
        strict_undefined=strict_undefined,
        raise_on_max_depth=raise_on_max_depth,
        normalize_date_values=normalize_date_values,
        extra_globals=extra_globals,
    ).resolved


def render_string(
    template_str: str,
    context: Mapping[str, Any] | None = None,
    *,
    strict_undefined: bool = True,
    normalize_date_values: bool = True,
    extra_globals: Mapping[str, Any] | None = None,
) -> Any:
    """Render a single Jinja2 string in memory with SandboxedEnvironment (preserves pure expression types)."""
    if not isinstance(template_str, str):
        raise TypeError("template_str must be a string.")
    if not _contains_jinja(template_str):
        return template_str
    env = _build_env(strict_undefined=strict_undefined, extra_globals=extra_globals)
    _parse_and_check(template_str, env, allow_statements=True)
    ctx = _normalize_dict_keys(context) if context else {}
    result, _, _ = _eval_single_template(
        template_str,
        ctx,
        env=env,
        strict_undefined=strict_undefined,
        normalize_date_values=normalize_date_values,
    )
    return result


def render_template_text(
    template_str: str,
    context: Mapping[str, Any] | None = None,
    *,
    strict_undefined: bool = True,
    extra_globals: Mapping[str, Any] | None = None,
) -> str:
    """Render a Jinja2 template text string in memory with SandboxedEnvironment, always returning str."""
    if not isinstance(template_str, str):
        raise TypeError("template_str must be a string.")
    if not _contains_jinja(template_str):
        return template_str
    env = _build_env(strict_undefined=strict_undefined, extra_globals=extra_globals)
    _parse_and_check(template_str, env, allow_statements=True)
    ctx = _normalize_dict_keys(context) if context else {}
    try:
        render_ctx, has_self_override = _prepare_render_ctx(ctx, env)
        template = _compile_template(
            template_str, env, has_self_override=has_self_override
        )
        rendered = template.render(render_ctx)
    except SecurityError as exc:
        raise YamgamSecurityError(str(exc)) from exc
    except UndefinedError as exc:
        if strict_undefined:
            raise YamgamExpressionError(f"Undefined variable: {exc}") from exc
        return ""
    except TemplateSyntaxError as exc:
        raise YamgamExpressionError(f"Template syntax error: {exc}") from exc
    except (ArithmeticError, TypeError, ValueError, LookupError) as exc:
        if strict_undefined:
            raise YamgamExpressionError(f"Expression evaluation error: {exc}") from exc
        return ""
    return rendered
