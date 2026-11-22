from __future__ import annotations

from collections.abc import Mapping
import copy
import math
import re
import uuid
from typing import Any

from box import Box
from jinja2 import TemplateSyntaxError, Undefined, exceptions, nodes
from jinja2.exceptions import SecurityError
from jinja2.parser import Parser
from jinja2.sandbox import SandboxedEnvironment
import yaml

from yamgam.state import state

MAX_SANDBOX_RANGE_ITEMS = 10_000
MAX_SANDBOX_REPEAT_LEN = 100_000
MAX_POW_EXPONENT = 1000
_MAX_INT_LITERAL = 10**4000
_MAX_INT_BITS = _MAX_INT_LITERAL.bit_length()

BLOCKED_ATTR_NAMES: frozenset[str] = frozenset(
    {
        "gi_frame",
        "gi_code",
        "cr_frame",
        "cr_code",
        "ag_frame",
        "ag_code",
        "f_locals",
        "f_globals",
        "f_builtins",
        "f_back",
        "func_globals",
        "func_code",
        "format",
        "format_map",
    }
)

BLOCKED_CALLABLE_NAMES: frozenset[str] = frozenset(
    {
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "breakpoint",
        "input",
        "type",
        "object",
    }
)

BLOCKED_CALLABLE_OBJECTS: frozenset[Any] = frozenset(
    {
        getattr,
        setattr,
        delattr,
        eval,
        exec,
        compile,
        open,
        __import__,
        type,
        object,
        vars,
        globals,
        locals,
        dir,
        breakpoint,
        input,
    }
)

_ATTR_INSPECTING_FILTERS: frozenset[str] = frozenset(
    {"attr", "groupby", "join", "selectattr", "rejectattr", "sort", "unique", "min", "max", "sum"}
)


def _safe_range(*args: Any) -> range:
    """Bounded replacement for built-in range() preventing CPU/memory DoS."""
    r = range(*args)
    if len(r) > MAX_SANDBOX_RANGE_ITEMS:
        raise SecurityError(
            f"range() size {len(r)} exceeds maximum allowed limit of {MAX_SANDBOX_RANGE_ITEMS}."
        )
    return r


def _guarded_attr_filter(obj: Any, name: str) -> Any:
    """Guarded replacement for Jinja2 `attr` filter rejecting private/dunder attributes."""
    if not isinstance(name, str) or not name:
        raise SecurityError(f"Access to restricted attribute {name!r} is forbidden.")
    for part in name.split("."):
        if part.startswith("_") or part in BLOCKED_ATTR_NAMES:
            raise SecurityError(f"Access to restricted attribute {name!r} is forbidden.")
    if isinstance(obj, Mapping) and name in obj:
        return obj[name]
    val = getattr(obj, name)
    if callable(val) and any(val is blocked for blocked in BLOCKED_CALLABLE_OBJECTS):
        raise SecurityError(f"Access to restricted attribute {name!r} is forbidden.")
    return val


def _validate_filter_attribute_expr(
    expr: nodes.Node,
    filter_name: str,
    *,
    allow_none: bool = False,
) -> None:
    """Validate that a filter attribute argument is a safe static string (no leading `_`)."""
    if not isinstance(expr, nodes.Const):
        raise SecurityError(f"Dynamic attribute argument on '{filter_name}' filter is forbidden.")
    val = expr.value
    if val is None and allow_none:
        return
    if not isinstance(val, str) or not val:
        raise SecurityError(
            f"Invalid attribute argument {val!r} on '{filter_name}' filter."
        )
    for part in val.split("."):
        if part.startswith("_") or part in BLOCKED_ATTR_NAMES:
            raise SecurityError(
                f"Access to restricted attribute {val!r} via '{filter_name}' filter is forbidden."
            )


def _validate_positional_filter_attrs(
    filter_name: str,
    args: list[nodes.Expr],
    *,
    label: str | None = None,
) -> None:
    """Validate positional attribute arguments for attribute-inspecting Jinja2 filters."""
    name_label = label or filter_name
    if filter_name == "attr":
        if not args:
            raise SecurityError(f"Dynamic '{name_label}' filter arguments are forbidden.")
        _validate_filter_attribute_expr(args[0], name_label)
    elif filter_name in {"groupby", "selectattr", "rejectattr"}:
        if args:
            _validate_filter_attribute_expr(args[0], name_label)
    elif filter_name == "sum":
        if args:
            _validate_filter_attribute_expr(args[0], name_label, allow_none=True)
    elif filter_name == "sort":
        if len(args) >= 3:
            _validate_filter_attribute_expr(args[2], name_label, allow_none=True)
    elif filter_name in {"unique", "min", "max", "join"}:
        if len(args) >= 2:
            _validate_filter_attribute_expr(args[1], name_label, allow_none=True)


def _is_nan_or_inf(val: int | float) -> bool:
    return isinstance(val, float) and (math.isnan(val) or math.isinf(val))


def _validate_pow_bounds(left: Any, right: Any) -> None:
    if isinstance(right, (int, float)) and not isinstance(right, bool):
        if _is_nan_or_inf(right) or abs(right) > MAX_POW_EXPONENT:
            raise SecurityError(
                f"Power exponent exceeds maximum allowed limit ({MAX_POW_EXPONENT})."
            )
        if isinstance(left, (int, float)) and not isinstance(left, bool):
            if isinstance(left, int) and isinstance(right, int) and right > 0:
                if abs(left) > 1 and left.bit_length() * right > _MAX_INT_BITS:
                    raise SecurityError(
                        "Power expression exceeds maximum allowed integer size."
                    )
            elif abs(left) > 1 and right > 0:
                try:
                    if _is_nan_or_inf(left) or math.log2(abs(left)) * right > 1023:
                        raise SecurityError(
                            "Power expression exceeds maximum allowed float range."
                        )
                except (ValueError, OverflowError) as exc:
                    raise SecurityError(
                        "Power expression exceeds maximum allowed float range."
                    ) from exc


def _eval_const_number(node: nodes.Node) -> int | float | None:
    """Safely evaluate a constant numeric AST subtree without risking DoS."""
    if isinstance(node, nodes.Const):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        return None
    if isinstance(node, nodes.Neg):
        val = _eval_const_number(node.node)
        return -val if val is not None else None
    if isinstance(node, nodes.Pos):
        val = _eval_const_number(node.node)
        return +val if val is not None else None
    if isinstance(node, nodes.Add):
        left = _eval_const_number(node.left)
        right = _eval_const_number(node.right)
        if left is not None and right is not None:
            try:
                return left + right
            except OverflowError as exc:
                raise SecurityError("Constant arithmetic expression overflows.") from exc
        return None
    if isinstance(node, nodes.Sub):
        left = _eval_const_number(node.left)
        right = _eval_const_number(node.right)
        if left is not None and right is not None:
            try:
                return left - right
            except OverflowError as exc:
                raise SecurityError("Constant arithmetic expression overflows.") from exc
        return None
    if isinstance(node, nodes.Mul):
        left = _eval_const_number(node.left)
        right = _eval_const_number(node.right)
        if left is not None and right is not None:
            if isinstance(left, int) and isinstance(right, int):
                if (
                    left != 0
                    and right != 0
                    and left.bit_length() + right.bit_length() - 1 > _MAX_INT_BITS
                ):
                    raise SecurityError(
                        "Multiplication expression exceeds maximum allowed integer size."
                    )
            try:
                result = left * right
            except OverflowError as exc:
                raise SecurityError("Constant arithmetic expression overflows.") from exc
            if isinstance(result, int) and result.bit_length() > _MAX_INT_BITS:
                raise SecurityError(
                    "Multiplication expression exceeds maximum allowed integer size."
                )
            return result
        return None
    if isinstance(node, nodes.Div):
        left = _eval_const_number(node.left)
        right = _eval_const_number(node.right)
        if left is not None and right is not None and right != 0:
            try:
                return left / right
            except OverflowError as exc:
                raise SecurityError("Constant arithmetic expression overflows.") from exc
        return None
    if isinstance(node, nodes.FloorDiv):
        left = _eval_const_number(node.left)
        right = _eval_const_number(node.right)
        if left is not None and right is not None and right != 0:
            try:
                return left // right
            except OverflowError as exc:
                raise SecurityError("Constant arithmetic expression overflows.") from exc
        return None
    if isinstance(node, nodes.Mod):
        left = _eval_const_number(node.left)
        right = _eval_const_number(node.right)
        if left is not None and right is not None and right != 0:
            return left % right
        return None
    if isinstance(node, nodes.Pow):
        left = _eval_const_number(node.left)
        right = _eval_const_number(node.right)
        if right is not None:
            _validate_pow_bounds(left, right)
        if left is not None and right is not None:
            if left == 0 and right < 0:
                return None
            try:
                return left**right
            except OverflowError as exc:
                raise SecurityError("Power expression overflows maximum float range.") from exc
        return None
    return None


def _eval_const_seq_len(node: nodes.Node) -> int | None:
    """Safely estimate the length of a constant string/bytes/list/tuple expression without allocating."""
    if isinstance(node, nodes.Const):
        if isinstance(node.value, (str, bytes, bytearray)):
            return len(node.value)
        return None
    if isinstance(node, (nodes.List, nodes.Tuple)):
        return len(node.items)
    if isinstance(node, nodes.Add):
        left_len = _eval_const_seq_len(node.left)
        right_len = _eval_const_seq_len(node.right)
        if left_len is not None and right_len is not None:
            total = left_len + right_len
            if total > MAX_SANDBOX_REPEAT_LEN:
                raise SecurityError(
                    f"Sequence size {total} exceeds maximum allowed limit ({MAX_SANDBOX_REPEAT_LEN})."
                )
            return total
        return None
    if isinstance(node, nodes.Concat):
        total = 0
        for child in node.nodes:
            child_len = _eval_const_seq_len(child)
            if child_len is None:
                return None
            total += child_len
            if total > MAX_SANDBOX_REPEAT_LEN:
                raise SecurityError(
                    f"Sequence size {total} exceeds maximum allowed limit ({MAX_SANDBOX_REPEAT_LEN})."
                )
        return total
    if isinstance(node, nodes.Mul):
        right_num = _eval_const_number(node.right)
        if isinstance(right_num, int):
            left_len = _eval_const_seq_len(node.left)
            if left_len is not None:
                if right_num <= 0 or left_len == 0:
                    return 0
                total = left_len * right_num
                if total > MAX_SANDBOX_REPEAT_LEN:
                    raise SecurityError(
                        f"Sequence repetition size {total} exceeds maximum allowed limit ({MAX_SANDBOX_REPEAT_LEN})."
                    )
                return total
        left_num = _eval_const_number(node.left)
        if isinstance(left_num, int):
            right_len = _eval_const_seq_len(node.right)
            if right_len is not None:
                if left_num <= 0 or right_len == 0:
                    return 0
                total = right_len * left_num
                if total > MAX_SANDBOX_REPEAT_LEN:
                    raise SecurityError(
                        f"Sequence repetition size {total} exceeds maximum allowed limit ({MAX_SANDBOX_REPEAT_LEN})."
                    )
                return total
        return None
    return None


def _extract_target_names(target: nodes.Node) -> set[str]:
    names: set[str] = set()
    if isinstance(target, nodes.Name):
        names.add(target.name)
    elif isinstance(target, nodes.Tuple):
        for item in target.items:
            names.update(_extract_target_names(item))
    return names


def check_jinja_ast_safety(parsed_ast: nodes.Node) -> None:
    """Statically inspect a Jinja2 AST to block SSTI even in unexecuted branches."""
    for node in parsed_ast.find_all(
        (
            nodes.Import,
            nodes.FromImport,
            nodes.Include,
            nodes.Extends,
            nodes.Macro,
            nodes.CallBlock,
            nodes.Assign,
            nodes.AssignBlock,
            nodes.For,
            nodes.Getattr,
            nodes.Getitem,
            nodes.Name,
            nodes.Filter,
            nodes.Const,
            nodes.Pow,
            nodes.Mul,
        )
    ):
        if isinstance(
            node,
            (
                nodes.Import,
                nodes.FromImport,
                nodes.Include,
                nodes.Extends,
                nodes.Macro,
                nodes.CallBlock,
            ),
        ):
            raise SecurityError(
                f"Jinja2 statement '{type(node).__name__}' is disallowed in yamgam templates."
            )
        if isinstance(node, (nodes.Assign, nodes.AssignBlock)):
            for target_name in _extract_target_names(node.target):
                if target_name.startswith("_") or target_name in BLOCKED_CALLABLE_NAMES:
                    raise SecurityError(
                        f"Assignment to restricted identifier '{target_name}' is forbidden."
                    )
        if isinstance(node, nodes.For):
            for target_name in _extract_target_names(node.target):
                if target_name.startswith("_") or target_name in BLOCKED_CALLABLE_NAMES:
                    raise SecurityError(
                        f"Loop variable '{target_name}' is restricted or forbidden."
                    )
        if isinstance(node, nodes.Const):
            if (
                isinstance(node.value, int)
                and not isinstance(node.value, bool)
                and abs(node.value) >= _MAX_INT_LITERAL
            ):
                raise SecurityError(
                    "Integer literal exceeds maximum allowed size (4000 digits)."
                )
        if isinstance(node, nodes.Pow):
            right_val = _eval_const_number(node.right)
            if right_val is not None:
                left_val = _eval_const_number(node.left)
                _validate_pow_bounds(left_val, right_val)
                cur: nodes.Node = node.left
                eff_exp: int | float = right_val
                while isinstance(cur, nodes.Pow):
                    inner_exp = _eval_const_number(cur.right)
                    if inner_exp is None:
                        break
                    eff_exp = eff_exp * inner_exp
                    if _is_nan_or_inf(eff_exp) or abs(eff_exp) > MAX_POW_EXPONENT:
                        raise SecurityError(
                            f"Chained power exponent exceeds maximum allowed limit ({MAX_POW_EXPONENT})."
                        )
                    cur = cur.left
            else:
                _eval_const_number(node.left)
        if isinstance(node, nodes.Mul):
            _eval_const_number(node)
            _eval_const_seq_len(node)
        if isinstance(node, nodes.Getattr):
            if node.attr.startswith("_") or node.attr in BLOCKED_ATTR_NAMES:
                raise SecurityError(
                    f"Access to private or restricted attribute '{node.attr}' is forbidden."
                )
        if isinstance(node, nodes.Getitem):
            if isinstance(node.arg, nodes.Const) and isinstance(node.arg.value, str):
                if node.arg.value.startswith("_") or node.arg.value in BLOCKED_ATTR_NAMES:
                    raise SecurityError(
                        f"Access to private or restricted key '{node.arg.value}' is forbidden."
                    )
        if isinstance(node, nodes.Name):
            if node.name in BLOCKED_CALLABLE_NAMES or node.name.startswith("_"):
                raise SecurityError(
                    f"Identifier '{node.name}' is forbidden in yamgam templates."
                )
        if isinstance(node, nodes.Filter):
            if (
                (node.dyn_args is not None or node.dyn_kwargs is not None)
                and (node.name in _ATTR_INSPECTING_FILTERS or node.name == "map")
            ):
                raise SecurityError(
                    f"Dynamic unpacking (*args/**kwargs) on '{node.name}' filter is forbidden."
                )

            for kw in node.kwargs:
                if kw.key == "attribute":
                    _validate_filter_attribute_expr(kw.value, node.name, allow_none=True)
                elif isinstance(kw.value, nodes.Const) and isinstance(kw.value.value, str):
                    if kw.value.value.startswith("_") or kw.value.value in BLOCKED_ATTR_NAMES:
                        raise SecurityError(
                            f"Restricted keyword value '{kw.value.value}' on '{node.name}' filter is forbidden."
                        )

            if node.name == "map" and node.args:
                if not isinstance(node.args[0], nodes.Const) or not isinstance(node.args[0].value, str):
                    raise SecurityError("Dynamic filter in 'map' is forbidden.")
                sub_filter = node.args[0].value
                if sub_filter in {"attr", "map"}:
                    raise SecurityError(f"'{sub_filter}' filter in 'map' is forbidden.")
                if sub_filter in _ATTR_INSPECTING_FILTERS:
                    _validate_positional_filter_attrs(
                        sub_filter,
                        node.args[1:],
                        label=f"map('{sub_filter}')",
                    )
                    for extra in node.args[1:]:
                        if isinstance(extra, nodes.Const) and isinstance(extra.value, str):
                            for part in extra.value.split("."):
                                if part.startswith("_") or part in BLOCKED_ATTR_NAMES:
                                    raise SecurityError(
                                        f"Access to restricted attribute '{extra.value}' via 'map({sub_filter!r})' filter is forbidden."
                                    )
            elif node.name in _ATTR_INSPECTING_FILTERS:
                _validate_positional_filter_attrs(node.name, node.args)


class YamgamSandboxEnvironment(SandboxedEnvironment):
    """Hardened Jinja2 SandboxedEnvironment blocking SSTI, dunder traversal, and unsafe callables."""

    intercepted_binops = frozenset({"**", "*"})

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.filters["attr"] = _guarded_attr_filter
        self.globals["range"] = _safe_range

    def call_binop(self, context: Any, operator: str, left: Any, right: Any) -> Any:
        if operator == "**":
            if isinstance(right, (int, float)) and not isinstance(right, bool):
                _validate_pow_bounds(left, right)
            if isinstance(left, Undefined) or isinstance(right, Undefined):
                return super().call_binop(context, operator, left, right)
            try:
                return super().call_binop(context, operator, left, right)
            except OverflowError as exc:
                raise SecurityError(
                    "Power expression overflows maximum float range."
                ) from exc
        if operator == "*":
            if isinstance(left, Undefined) or isinstance(right, Undefined):
                return super().call_binop(context, operator, left, right)
            if isinstance(left, int) and isinstance(right, int):
                if (
                    left != 0
                    and right != 0
                    and (left.bit_length() + right.bit_length() - 1) > _MAX_INT_BITS
                ):
                    raise SecurityError(
                        "Multiplication expression exceeds maximum allowed integer size."
                    )
                try:
                    result = super().call_binop(context, operator, left, right)
                except OverflowError as exc:
                    raise SecurityError(
                        "Multiplication expression overflows."
                    ) from exc
                if isinstance(result, int) and result.bit_length() > _MAX_INT_BITS:
                    raise SecurityError(
                        "Multiplication expression exceeds maximum allowed integer size."
                    )
                return result
            seq: Any = None
            repeat_count: int | None = None
            if isinstance(left, (str, bytes, bytearray, list, tuple)) and isinstance(right, int):
                seq, repeat_count = left, right
            elif isinstance(right, (str, bytes, bytearray, list, tuple)) and isinstance(left, int):
                seq, repeat_count = right, left
            if repeat_count is not None and repeat_count > 0:
                seq_len = len(seq)
                if seq_len > 0 and seq_len * repeat_count > MAX_SANDBOX_REPEAT_LEN:
                    raise SecurityError(
                        f"Sequence repetition size {seq_len * repeat_count} exceeds maximum allowed limit ({MAX_SANDBOX_REPEAT_LEN})."
                    )
            try:
                return super().call_binop(context, operator, left, right)
            except OverflowError as exc:
                raise SecurityError(
                    "Multiplication expression overflows."
                ) from exc
        return super().call_binop(context, operator, left, right)

    binop = call_binop

    def _parse(self, source: str, name: str | None, filename: str | None) -> nodes.Template:
        try:
            parsed = super()._parse(source, name, filename)
        except (ValueError, ArithmeticError, OverflowError) as exc:
            raise SecurityError(
                f"Unsafe or overflowing constant expression in template: {exc}"
            ) from exc
        check_jinja_ast_safety(parsed)
        return parsed

    def compile_expression(
        self, source: str, undefined_to_none: bool = True
    ) -> Any:
        parser = Parser(self, source, state="variable")
        try:
            expr = parser.parse_expression()
            if not parser.stream.eos:
                raise TemplateSyntaxError(
                    "chunk after expression", parser.stream.current.lineno, None, None
                )
            expr.set_environment(self)
        except TemplateSyntaxError:
            self.handle_exception(source=source)
        except (ValueError, ArithmeticError, OverflowError) as exc:
            raise SecurityError(
                f"Unsafe or overflowing constant expression in template: {exc}"
            ) from exc

        check_jinja_ast_safety(nodes.Template([nodes.Output([expr], lineno=1)], lineno=1))
        body = [nodes.Assign(nodes.Name("result", "store"), expr, lineno=1)]
        template = self.from_string(nodes.Template(body, lineno=1))
        from jinja2.environment import TemplateExpression

        return TemplateExpression(template, undefined_to_none)

    def unsafe_undefined(self, obj: Any, attribute: str) -> Undefined:
        raise SecurityError(
            f"Access to attribute {attribute!r} of {type(obj).__name__!r} object is unsafe."
        )

    def getattr(self, obj: Any, attribute: str) -> Any:
        if attribute.startswith("_") or attribute in BLOCKED_ATTR_NAMES:
            raise SecurityError(
                f"Access to restricted attribute '{attribute}' is forbidden."
            )
        if isinstance(obj, Mapping) and attribute in obj:
            return obj[attribute]
        return super().getattr(obj, attribute)

    def getitem(self, obj: Any, argument: Any) -> Any:
        if isinstance(argument, str) and (
            argument.startswith("_") or argument in BLOCKED_ATTR_NAMES
        ):
            raise SecurityError(
                f"Access to restricted key '{argument}' is forbidden."
            )
        return super().getitem(obj, argument)

    def is_safe_attribute(self, obj: Any, attr: str, value: Any) -> bool:
        if attr.startswith("_") or attr in BLOCKED_ATTR_NAMES:
            return False
        return super().is_safe_attribute(obj, attr, value)

    def is_safe_callable(self, obj: Any) -> bool:
        if any(obj is blocked for blocked in BLOCKED_CALLABLE_OBJECTS):
            return False
        return super().is_safe_callable(obj)


# Custom Loader to merge instead of replace
class DeepMergeLoader(yaml.SafeLoader):
    def _recursive_merge(self, dict1: dict[Any, Any], dict2: dict[Any, Any]) -> dict[Any, Any]:
        """Recursively merges dict2 into a fresh copy of dict1 without mutating YAML anchors."""
        merged = dict(dict1)
        for key, value in dict2.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._recursive_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)

            if key in mapping and isinstance(mapping[key], dict) and isinstance(value, dict):
                # Recursively merge the dictionaries
                mapping[key] = self._recursive_merge(mapping[key], value)
            else:
                mapping[key] = value
        return mapping


# Register the constructor for all mappings
DeepMergeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    DeepMergeLoader.construct_mapping,
)


class NullUndefined(Undefined):
    UNDEFINED = f"UNDEFINED:{uuid.uuid4().hex}"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.parent_object = kwargs.get("obj")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        new_name = f"{self._undefined_name}.{name}" if self._undefined_name else name
        return NullUndefined(name=new_name)

    def __str__(self) -> str:
        return f"{NullUndefined.UNDEFINED}:{self._undefined_name}"

    def __getitem__(self, name: Any) -> Any:
        return self.__getattr__(str(name))

    def __iter__(self):
        return iter([])

    def __bool__(self) -> bool:
        return False

    def _passthrough(self, other: Any = None) -> "NullUndefined":
        return NullUndefined(name=self._undefined_name)

    __add__ = __radd__ = _passthrough
    __sub__ = __rsub__ = _passthrough
    __mul__ = __rmul__ = _passthrough
    __truediv__ = __rtruediv__ = _passthrough
    __floordiv__ = __rfloordiv__ = _passthrough
    __mod__ = __rmod__ = _passthrough
    __pow__ = __rpow__ = _passthrough
    __neg__ = __pos__ = _passthrough

    def __lt__(self, other: Any) -> bool:
        return False

    def __le__(self, other: Any) -> bool:
        return False

    def __gt__(self, other: Any) -> bool:
        return False

    def __ge__(self, other: Any) -> bool:
        return False

    def __int__(self) -> int:
        return 0

    def __float__(self) -> float:
        return 0.0


class _PPTXRendering:
    table_data_pattern = re.compile(r"\{\{\s*table\s*:\s*([^}\s]+)\s*\}\}")

    def __init__(self, input_path: str, data: Mapping[str, Any]):
        self.input_path = input_path
        self.model = data["model"] if isinstance(data, Mapping) and "model" in data else data

        self.env = YamgamSandboxEnvironment()

        self.current_slide = None
        self.current_shape = None

        self.message_raw: list[str] = []

    def process(self, output_path: str) -> str:
        import pptx

        ppt = pptx.Presentation(self.input_path)
        for slide in ppt.slides:
            self.current_slide = slide
            self._render_slide(slide)
        ppt.save(output_path)
        return "\n".join(self.message_raw)

    def _render_slide(self, slide: Any) -> None:
        for shape in slide.shapes:
            self.current_shape = shape
            self._render_shape(shape)

    def _render_shape(self, shape: Any) -> None:
        if shape.has_text_frame:
            self._render_text_frame(shape.text_frame)
        if shape.has_table:
            self._render_table(shape.table)

    def _render_text_frame(self, text_frame: Any) -> None:
        for paragraph in text_frame.paragraphs:
            self._render_paragraph(paragraph)

    def _merge_placeholder_runs(self, paragraph: Any) -> None:
        runs = paragraph.runs
        i = 0
        while i < len(runs):
            text = runs[i].text or ""
            if "{{" in text:
                merged = text
                j = i + 1
                while j < len(runs) and "}}" not in merged:
                    merged += runs[j].text or ""
                    j += 1
                if "}}" in merged:
                    runs[i].text = merged
                    for k in range(i + 1, j):
                        runs[k].text = ""
                    i = j
                else:
                    break
            else:
                i += 1

    def _render_paragraph(self, paragraph: Any) -> None:
        self._merge_placeholder_runs(paragraph)
        for run in paragraph.runs:
            self._render_run(run)

    def _render_table(self, table: Any) -> None:
        self._prepare_table(table)
        for cell in table.iter_cells():
            self._render_cell(cell)

    def _prepare_table(self, table: Any) -> None:
        for cell in table.iter_cells():
            matches = re.findall(self.table_data_pattern, cell.text)
            if matches:
                table_data_key = matches[0].strip()
                self._remove_row(table, 0)
                if table_data := self.model.get(table_data_key):
                    for i, _row in enumerate(table_data):
                        self._add_row(table, i, table_data_key)
                    self._remove_row(table, 1)
            break

    def _render_cell(self, cell: Any) -> None:
        self._render_text_frame(cell.text_frame)

    def _render_run(self, run: Any) -> None:
        try:
            template = self.env.from_string(str(run.text))
            rendered = template.render(self.model)
        except exceptions.UndefinedError as error:
            error_text = f"{error.__class__.__name__}: {error}"
            self.message_raw.append(error_text)
        except exceptions.TemplateSyntaxError as error:
            error_text = (
                f"{error.__class__.__name__}: {error}\n"
                "you should re-write the whole {{}} tag"
            )
            self.message_raw.append(error_text)
        else:
            run.text = rendered

    @staticmethod
    def _add_row(table: Any, row_index: int, table_data_key: str) -> None:
        import pptx

        new_row = copy.deepcopy(table._tbl.tr_lst[-1])

        for cell_index, tc in enumerate(new_row.tc_lst):
            cell = pptx.table._Cell(tc, new_row.tc_lst)
            for paragraph in cell.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.text = f"{{{{ {table_data_key}[{row_index}][{cell_index}] }}}}"

        table._tbl.append(new_row)

    @staticmethod
    def _remove_row(table: Any, row_index: int) -> None:
        row = list(table.rows)[row_index]
        table._tbl.remove(row._tr)


def xlsx_file(input_path: str, output_path: str | None, context: Any) -> None:
    import openpyxl

    wb = openpyxl.load_workbook(input_path)
    env = YamgamSandboxEnvironment()

    for sheet in wb.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                if isinstance(cell.value, str):
                    template = env.from_string(cell.value)
                    cell.value = template.render(context)

    if output_path:
        wb.save(output_path)


def docx_file(input_path: str, output_path: str | None, context: Any) -> None:
    import docxtpl

    doc = docxtpl.DocxTemplate(input_path)
    doc.render(context, jinja_env=YamgamSandboxEnvironment())
    if output_path:
        doc.save(output_path)


def pptx_file(input_path: str, output_path: str | None, context: Any) -> None:
    rendering = _PPTXRendering(input_path, context)
    if output_path:
        message = rendering.process(output_path)
        if state.verbose:
            print(message)


_NODE_PROPS_RE = r"(?:(?:&\S+|!\S*)\s+){0,2}"
_TRAILING_PROPS_RE = r"(?:&\S+|!\S*)(?:\s+(?:&\S+|!\S*))?"
_MAPPING_KEY_RE = (
    r"(?:\"(?:\\.|[^\"\\])*\"|'(?:''|[^'])*'|(?!-\s)[^:#\s&!|>\"'](?:[^:]|:(?!\s))*)"
)
_BLOCK_INDICATOR_RE = r"[|>](?:[+-](?P<ind1>[1-9])?|(?P<ind2>[1-9])[+-]?)?"

_DOC_MARKER_RE = re.compile(rf"^---(?:\s+{_TRAILING_PROPS_RE})?\s*$")
_DOC_END_RE = re.compile(r"^\.\.\.\s*$")
_BLOCK_SCALAR_START_RE = re.compile(
    r"^(?P<ws>\s*)"
    r"(?P<dash>(?:-\s+)*)"
    rf"(?P<key>{_NODE_PROPS_RE}{_MAPPING_KEY_RE}:\s+)?"
    rf"{_NODE_PROPS_RE}"
    rf"{_BLOCK_INDICATOR_RE}\s*$"
)
_DOC_BLOCK_SCALAR_START_RE = re.compile(
    rf"^---\s+{_NODE_PROPS_RE}{_BLOCK_INDICATOR_RE}\s*$"
)
_PENDING_KEY_RE = re.compile(
    r"^(?P<ws>\s*)"
    r"(?P<dash>(?:-\s+)*)"
    rf"{_NODE_PROPS_RE}{_MAPPING_KEY_RE}:(?:\s+{_TRAILING_PROPS_RE})?\s*$"
)
_PENDING_DASH_RE = re.compile(
    r"^(?P<ws>\s*)"
    r"(?P<dash>(?:-\s+)*-)"
    rf"(?:\s+{_TRAILING_PROPS_RE})?\s*$"
)
_PROPERTY_ONLY_RE = re.compile(
    r"^(?P<ws>\s*)(?:&\S+|!\S*)(?:\s+(?:&\S+|!\S*))?\s*$"
)


def _strip_yaml_comments_outside_blocks(raw: str) -> str:
    """Strip YAML `#` comments outside quotes, Jinja2 tags, and block scalars."""
    lines = raw.splitlines(keepends=True)
    out: list[str] = []
    in_block_scalar = False
    block_owner_indent = 0
    block_content_indent: int | None = None
    max_leading_blank_indent = 0
    pending_owner_indent: int | None = None
    can_start_root_block = True
    in_single_quote = False
    in_double_quote = False
    in_jinja_expr = False
    in_jinja_block = False
    in_jinja_comment = False
    jinja_quote: str | None = None

    for line in lines:
        if line.endswith("\r\n"):
            body, ending = line[:-2], "\r\n"
        elif line.endswith("\n"):
            body, ending = line[:-1], "\n"
        else:
            body, ending = line, ""

        if in_block_scalar:
            if not body.strip():
                if block_content_indent is None and body == " " * len(body):
                    max_leading_blank_indent = max(max_leading_blank_indent, len(body))
                out.append(line)
                continue
            indent = len(body) - len(body.lstrip(" \t"))
            if block_content_indent is not None:
                if indent >= block_content_indent:
                    out.append(line)
                    continue
            elif indent > block_owner_indent and indent >= max_leading_blank_indent:
                block_content_indent = indent
                out.append(line)
                continue
            in_block_scalar = False
            block_content_indent = None
            max_leading_blank_indent = 0

        started_in_multiline = (
            in_single_quote
            or in_double_quote
            or in_jinja_expr
            or in_jinja_block
            or in_jinja_comment
        )
        cleaned_chars: list[str] = []
        i = 0
        n = len(body)
        while i < n:
            ch = body[i]
            nxt = body[i : i + 2]

            if in_jinja_comment:
                cleaned_chars.append(ch)
                if nxt == "#}":
                    cleaned_chars.append("}")
                    in_jinja_comment = False
                    i += 2
                    continue
                i += 1
                continue

            if in_jinja_expr or in_jinja_block:
                cleaned_chars.append(ch)
                if jinja_quote is not None:
                    if ch == "\\" and i + 1 < n:
                        cleaned_chars.append(body[i + 1])
                        i += 2
                        continue
                    if ch == jinja_quote:
                        jinja_quote = None
                else:
                    if ch in ("'", '"'):
                        jinja_quote = ch
                    elif in_jinja_expr and nxt == "}}":
                        cleaned_chars.append("}")
                        in_jinja_expr = False
                        i += 2
                        continue
                    elif in_jinja_block and nxt == "%}":
                        cleaned_chars.append("}")
                        in_jinja_block = False
                        i += 2
                        continue
                i += 1
                continue

            if in_single_quote:
                cleaned_chars.append(ch)
                if ch == "'":
                    if i + 1 < n and body[i + 1] == "'":
                        cleaned_chars.append("'")
                        i += 2
                        continue
                    in_single_quote = False
                i += 1
                continue

            if in_double_quote:
                cleaned_chars.append(ch)
                if ch == "\\" and i + 1 < n:
                    cleaned_chars.append(body[i + 1])
                    i += 2
                    continue
                if ch == '"':
                    in_double_quote = False
                i += 1
                continue

            # Outside quotes and outside Jinja tags
            if nxt == "{#":
                in_jinja_comment = True
                cleaned_chars.append("{#")
                i += 2
                continue
            if nxt == "{{":
                in_jinja_expr = True
                cleaned_chars.append("{{")
                i += 2
                continue
            if nxt == "{%":
                in_jinja_block = True
                cleaned_chars.append("{%")
                i += 2
                continue
            if ch == "'":
                in_single_quote = True
                cleaned_chars.append(ch)
                i += 1
                continue
            if ch == '"':
                in_double_quote = True
                cleaned_chars.append(ch)
                i += 1
                continue
            if ch == "#" and (i == 0 or body[i - 1] in (" ", "\t")):
                # Genuine YAML comment to end of line
                break

            cleaned_chars.append(ch)
            i += 1

        cleaned_body = "".join(cleaned_chars)
        out.append(cleaned_body + ending)

        if (
            started_in_multiline
            or in_single_quote
            or in_double_quote
            or in_jinja_expr
            or in_jinja_block
            or in_jinja_comment
        ):
            pending_owner_indent = None
            can_start_root_block = False
            continue

        stripped_cleaned = cleaned_body.strip()
        if not stripped_cleaned:
            continue

        if _DOC_MARKER_RE.match(cleaned_body):
            pending_owner_indent = None
            can_start_root_block = True
            continue
        if _DOC_END_RE.match(cleaned_body):
            pending_owner_indent = None
            can_start_root_block = False
            continue

        doc_m = _DOC_BLOCK_SCALAR_START_RE.match(cleaned_body)
        if doc_m:
            explicit_str = doc_m.group("ind1") or doc_m.group("ind2")
            explicit_indent = int(explicit_str) if explicit_str else None
            in_block_scalar = True
            block_owner_indent = 0
            block_content_indent = explicit_indent
            max_leading_blank_indent = 0
            pending_owner_indent = None
            can_start_root_block = False
            continue

        m = _BLOCK_SCALAR_START_RE.match(cleaned_body)
        if m:
            ws = m.group("ws") or ""
            dash = m.group("dash") or ""
            has_key = m.group("key") is not None
            explicit_str = m.group("ind1") or m.group("ind2")
            explicit_indent = int(explicit_str) if explicit_str else None

            if has_key or dash:
                dash_indent = len(dash) if has_key else len(re.sub(r"-\s+$", "", dash))
                owner_indent = len(ws) + dash_indent
                in_block_scalar = True
                block_owner_indent = owner_indent
                block_content_indent = (
                    owner_indent + explicit_indent if explicit_indent is not None else None
                )
                max_leading_blank_indent = 0
            else:
                line_indent = len(ws)
                if pending_owner_indent is not None and line_indent >= pending_owner_indent:
                    in_block_scalar = True
                    block_owner_indent = pending_owner_indent
                    block_content_indent = (
                        pending_owner_indent + explicit_indent
                        if explicit_indent is not None
                        else None
                    )
                    max_leading_blank_indent = 0
                elif can_start_root_block:
                    in_block_scalar = True
                    block_owner_indent = 0
                    block_content_indent = explicit_indent
                    max_leading_blank_indent = 0

            pending_owner_indent = None
            can_start_root_block = False
            continue

        pk = _PENDING_KEY_RE.match(cleaned_body)
        if pk:
            pending_owner_indent = len(pk.group("ws") or "") + len(pk.group("dash") or "")
            can_start_root_block = False
            continue

        pd = _PENDING_DASH_RE.match(cleaned_body)
        if pd:
            pending_owner_indent = len(pd.group("ws") or "") + len(pd.group("dash") or "") - 1
            can_start_root_block = False
            continue

        prop_m = _PROPERTY_ONLY_RE.match(cleaned_body)
        if prop_m:
            if (
                pending_owner_indent is not None
                and len(prop_m.group("ws") or "") > pending_owner_indent
            ):
                can_start_root_block = False
                continue
            if pending_owner_indent is None and can_start_root_block:
                continue

        pending_owner_indent = None
        can_start_root_block = False

    return "".join(out)


def raw_yaml(raw: str, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    env = YamgamSandboxEnvironment(undefined=NullUndefined)
    cleaned_raw = _strip_yaml_comments_outside_blocks(raw)
    template = env.from_string(cleaned_raw)
    ctx_box = context if isinstance(context, Box) else Box(dict(context) if context else {})
    rendered = template.render(ctx_box)
    loaded = yaml.load(rendered, Loader=DeepMergeLoader)
    if loaded is None:
        return {}
    if isinstance(loaded, dict):
        return loaded
    return {}


def recursive_yaml(raw: str, depth: int = 10) -> Box:
    previous = None
    current: dict[str, Any] = {}
    count = 0
    while previous != current and count < depth:
        previous = current
        current = raw_yaml(raw, current)
        count += 1
    return Box(current)
