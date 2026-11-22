from __future__ import annotations

from pathlib import Path
import time
from types import MappingProxyType

from docx import Document as DocxDocument
from jinja2.exceptions import SecurityError
import pytest
from typer.testing import CliRunner
import yaml

from yamgam import (
    YamgamExpressionError,
    YamgamSecurityError,
    YamgamYamlError,
    deep_merge,
    extract_expressions,
    extract_referenced_variables,
    parse_yaml_layer,
    raw_yaml,
    recursive_yaml,
    render_string,
    render_template_text,
    resolve_context,
    validate_expressions,
)
from yamgam.api import _ref_segments_has_unresolved_jinja
from yamgam.cli import app as cli_app
from yamgam.render import _strip_yaml_comments_outside_blocks, docx_file


# 1. Pow DoS blocked at both AST and runtime
@pytest.mark.parametrize(
    "expr",
    [
        "{{ 2 ** (500 + 600) }}",
        "{{ 2 ** (40 * 30) }}",
        "{{ (10 ** 500) ** 3 }}",
        "{{ 2 ** (-(-1001)) }}",
        "{{ (2 ** 500) ** 500 }}",
        "{{ 2 ** (10 ** 500) }}",
    ],
)
def test_pow_dos_blocked_statically_across_all_entrypoints(expr: str, tmp_path: Path) -> None:
    with pytest.raises((SecurityError, YamgamSecurityError)):
        validate_expressions({"x": expr})
    with pytest.raises((SecurityError, YamgamSecurityError)):
        resolve_context({"x": expr})
    with pytest.raises((SecurityError, YamgamSecurityError)):
        render_string(expr)
    with pytest.raises((SecurityError, YamgamSecurityError)):
        render_template_text(expr)
    with pytest.raises((SecurityError, YamgamSecurityError)):
        raw_yaml(f"x: '{expr}'", {})

    docx_in = tmp_path / "pow.docx"
    docx_out = tmp_path / "pow_out.docx"
    doc = DocxDocument()
    doc.add_paragraph(expr)
    doc.save(docx_in)
    with pytest.raises(SecurityError):
        docx_file(str(docx_in), str(docx_out), {})


def test_pow_dos_blocked_at_runtime_with_context_variables() -> None:
    for ctx in [
        {"a": 2, "b": 1001},
        {"a": 10**500, "b": 10},
        {"a": 1e100, "b": 10},
    ]:
        with pytest.raises((SecurityError, YamgamSecurityError)):
            render_string("{{ a ** b }}", ctx)
        with pytest.raises((SecurityError, YamgamSecurityError)):
            render_template_text("{{ a ** b }}", ctx)
        with pytest.raises((SecurityError, YamgamSecurityError)):
            resolve_context({**ctx, "out": "{{ a ** b }}"})

    # Safe powers still work
    assert render_string("{{ (2 ** 3) ** 4 }}") == 4096
    assert render_string("{{ a ** b }}", {"a": 2, "b": 10}) == 1024


# 2. Whitespace-control delimiters {{- ... -}} without corrupting unary minus
def test_whitespace_control_pure_expressions_and_unary_minus() -> None:
    assert render_string("{{- 5 }}") == 5
    assert render_string("{{ -5 }}") == -5
    assert render_string("{{- -5 }}") == -5
    assert render_string("{{ 5 -}}") == 5
    assert render_string("{{- 1 + 1 -}}") == 2
    assert render_string("{{+ 7 }}") == 7
    assert render_string("{{- 'abc' -}}") == "abc"
    assert render_string("  {{- 5 -}}  ") == 5

    res = resolve_context({"w": 42, "a": "{{- 1 + 1 -}}", "b": "{{- w -}}", "c": "{{ -w }}"})
    assert res.converged is True
    assert res.resolved["a"] == 2
    assert res.resolved["b"] == 42
    assert res.resolved["c"] == -42


# 3. Unhashable callable does not crash sandbox
def test_unhashable_callable_allowed_cleanly() -> None:
    class UnhashableCallable:
        __hash__ = None  # type: ignore[assignment]

        def __call__(self) -> str:
            return "ok"

    class Holder:
        fn = UnhashableCallable()

    assert render_string("{{ fn() }}", {"fn": UnhashableCallable()}) == "ok"
    assert render_string("{{ obj | attr('fn') }}", {"obj": Holder()})() == "ok"
    assert render_template_text("{{ fn() }}", {"fn": UnhashableCallable()}) == "ok"


# 4. Context key "self" does not collide with Jinja2 Template.render / TemplateExpression
def test_context_key_self_works_across_apis() -> None:
    assert render_string("{{ self }}", {"self": 20}) == 20
    assert render_string("{{ result + self }}", {"result": 10, "self": 20}) == 30
    assert render_template_text("Hi {{ self }}", {"self": "World"}) == "Hi World"

    res = resolve_context({"result": 10, "self": 20, "x": "{{ result + self }}", "msg": "Hi {{ self }}"})
    assert res.converged is True
    assert res.resolved["x"] == 30
    assert res.resolved["msg"] == "Hi 20"

    assert render_template_text("{{ self }}", {}, extra_globals={"self": "Global"}) == "Global"
    assert resolve_context({"msg": "{{ self }}"}, extra_globals={"self": "Global"}).resolved["msg"] == "Global"


# 5. Context keys shadowing Jinja2 globals (e.g. range, dict)
def test_context_key_shadowing_jinja_global_resolves_in_order() -> None:
    res = resolve_context({"range": "{{ base + 1 }}", "base": 10, "doubled": "{{ range * 2 }}"})
    assert res.converged is True
    assert res.resolved["range"] == 11
    assert isinstance(res.resolved["range"], int)
    assert res.resolved["doubled"] == 22
    assert isinstance(res.resolved["doubled"], int)


# 6. Dotted/special dict keys in dependency tracking
def test_dotted_dict_keys_deferred_properly() -> None:
    res = resolve_context(
        {"d": {"a.b": "{{ base + 1 }}"}, "base": 10, "calc": "{{ d['a.b'] * 2 }}"}
    )
    assert res.converged is True
    assert res.resolved["d"]["a.b"] == 11
    assert res.resolved["calc"] == 22
    assert isinstance(res.resolved["calc"], int)


# 7 & 8. ResolutionResult.converged and runtime arithmetic/type errors + chained non-strict undefineds
def test_converged_flag_and_runtime_errors_and_chained_undefineds() -> None:
    r_undef = resolve_context(
        {"a": "Hello {{ missing_var }}", "b": "{{ missing_pure }}"},
        strict_undefined=False,
        raise_on_max_depth=False,
    )
    assert r_undef.converged is False
    assert set(r_undef.unresolved_paths) == {"a", "b"}
    assert "a" in r_undef.errors and "b" in r_undef.errors

    # Chained expression on None produced by non-strict undefined must not crash
    r_chained = resolve_context(
        {"a": "{{ missing }}", "b": "{{ a + 1 }}"},
        strict_undefined=False,
        raise_on_max_depth=False,
    )
    assert r_chained.converged is False
    assert r_chained.resolved == {"a": None, "b": None}
    assert "a" in r_chained.errors
    assert "b" in r_chained.errors
    assert set(r_chained.unresolved_paths) == {"a", "b"}

    # Strict mode wraps runtime arithmetic/type/value/lookup errors in YamgamExpressionError
    for expr in ["{{ 1 / 0 }}", "{{ 'a' + 1 }}", "{{ [1][5] }}"]:
        with pytest.raises(YamgamExpressionError):
            render_string(expr, {})
        with pytest.raises(YamgamExpressionError):
            render_template_text(expr, {})
        with pytest.raises(YamgamExpressionError) as exc_info:
            resolve_context({"bad": expr})
        assert exc_info.value.path == "bad"


# 9. Prevent {% raw %} and escaped-Jinja double evaluation
def test_raw_and_escaped_jinja_not_double_evaluated() -> None:
    r1 = resolve_context({"x": 42, "a": "{% raw %}{{ x }}{% endraw %}", "b": "val={{ a }}"})
    assert r1.converged is True
    assert r1.resolved["a"] == "{{ x }}"
    assert r1.resolved["b"] == "val={{ x }}"

    r2 = resolve_context({"a": "{% raw %}{{ not_eval }}{% endraw %}"})
    assert r2.converged is True
    assert r2.resolved["a"] == "{{ not_eval }}"

    r3 = resolve_context({"a": "{% raw %}{{ unclosed {% endraw %}"})
    assert r3.converged is True
    assert r3.resolved["a"] == "{{ unclosed "

    r4 = resolve_context({"x": 99, "a": "{{ '{{ x }}' }}"})
    assert r4.converged is True
    assert r4.resolved["a"] == "{{ x }}"


# 10. DeepMergeLoader <<: *alias merge keys and raw_yaml comment safety
def test_yaml_merge_keys_and_raw_yaml_comments() -> None:
    yaml_with_merge = """
defs: &defs
  color: "#1B365D"
  nested:
    a: 1
    b: 2
app:
  <<: *defs
  nested:
    b: 99
"""
    parsed = parse_yaml_layer(yaml_with_merge)
    assert parsed["app"]["color"] == "#1B365D"
    assert parsed["app"]["nested"] == {"a": 1, "b": 99}
    # Anchor source must not be mutated
    assert parsed["defs"]["nested"] == {"a": 1, "b": 2}

    rendered = raw_yaml(yaml_with_merge, {})
    assert rendered["app"]["color"] == "#1B365D"
    assert rendered["app"]["nested"] == {"a": 1, "b": 99}
    assert rendered["defs"]["nested"] == {"a": 1, "b": 2}

    # Standalone and inline YAML comments containing invalid/unsafe Jinja2 must not crash raw_yaml/recursive_yaml
    commented_yaml = """
# {{ broken syntax
   # {{ "".__class__ }}
color: "#1B365D" # {{ invalid_comment_jinja
url: https://example.com/docs#section-2
block: |
  # Heading inside block scalar {{ literal }}
  Item #7
"""
    r_raw = raw_yaml(commented_yaml, {"literal": "ok"})
    assert r_raw["color"] == "#1B365D"
    assert r_raw["url"] == "https://example.com/docs#section-2"
    assert "# Heading inside block scalar ok" in r_raw["block"]
    assert "Item #7" in r_raw["block"]

    r_rec = recursive_yaml(commented_yaml)
    assert r_rec["color"] == "#1B365D"


# 11a. parse_yaml_layer wraps unhashable key TypeError in YamgamYamlError
def test_parse_yaml_layer_wraps_unhashable_key_typeerror() -> None:
    with pytest.raises(YamgamYamlError):
        parse_yaml_layer("a: {{ b }}\n")


# 11b. extract_referenced_variables respects lexical For/Assign scope
def test_extract_referenced_variables_lexical_scoping() -> None:
    refs = extract_referenced_variables("{% for x in rows %}{{ x.name }}{% endfor %} {{ x.other }}")
    assert refs == ["rows", "x.other"]

    refs2 = extract_referenced_variables("{% for x in x %}{{ x.name }}{% endfor %}")
    assert refs2 == ["x"]

    refs3 = extract_referenced_variables("{{ y }} {% set y = 1 %}{{ y }}")
    assert refs3 == ["y"]


# 11c. extract_expressions rejects unclosed Jinja2 tags
@pytest.mark.parametrize("bad", ["{{ foo", "{% if x", "{# comment"])
def test_extract_expressions_rejects_unclosed_tags(bad: str) -> None:
    with pytest.raises(YamgamExpressionError):
        extract_expressions(bad)


# 11d. CLI works without config: block when -p is passed, and errors cleanly without -p/-i
def test_cli_without_config_block(tmp_path: Path) -> None:
    plain = tmp_path / "plain.yaml"
    plain.write_text('name: "DocTag"\ncolor: "#1B365D"\n', encoding="utf-8")

    runner = CliRunner()
    res_print = runner.invoke(cli_app, [str(plain), "-p", "json"])
    assert res_print.exit_code == 0, res_print.output
    assert '"name": "DocTag"' in res_print.output
    assert '"color": "#1B365D"' in res_print.output

    res_no_mode = runner.invoke(cli_app, [str(plain)])
    assert res_no_mode.exit_code == 1
    assert "No config found" in res_no_mode.output


# 11e. deep_merge handles MappingProxyType inside lists
def test_deep_merge_handles_mappingproxy_inside_lists() -> None:
    merged = deep_merge(
        {"arr": [MappingProxyType({1: "one"})]},
        {"arr2": [MappingProxyType({2: "two"})]},
    )
    assert merged == {"arr": [{"1": "one"}], "arr2": [{"2": "two"}]}


# 12. Iteration 3 regressions: YAML 1.2 block scalar headers & negative index segments
@pytest.mark.parametrize(
    "indicator",
    [
        "|",
        ">",
        "|-",
        "|+",
        ">-",
        ">+",
        "|2",
        "|-2",
        "|2-",
        "|+2",
        "|2+",
        ">-2",
        ">2-",
        ">+2",
        ">2+",
    ],
)
def test_block_scalar_headers_preserve_hashes_and_strip_trailing_comments(indicator: str) -> None:
    yml = (
        f"section:\n"
        f"  doc: {indicator} # header comment\n"
        f"    # Heading 1\n"
        f"    Line #2 {{{{ val }}}}\n"
        f"  other: 1 # subsequent comment\n"
    )
    stripped = _strip_yaml_comments_outside_blocks(yml)
    assert "# header comment" not in stripped
    assert "# subsequent comment" not in stripped
    assert "# Heading 1" in stripped
    assert "Line #2" in stripped

    res_raw = raw_yaml(yml, {"val": "ok"})
    assert "# Heading 1" in res_raw["section"]["doc"]
    assert "Line #2 ok" in res_raw["section"]["doc"]
    assert res_raw["section"]["other"] == 1

    yml_rec = (
        f"val: ok\n"
        f"section:\n"
        f"  doc: {indicator} # header comment\n"
        f"    # Heading 1\n"
        f"    Line #2 {{{{ val }}}}\n"
        f"  other: 1 # subsequent comment\n"
    )
    res_rec = recursive_yaml(yml_rec)
    assert "# Heading 1" in res_rec["section"]["doc"]
    assert "Line #2 ok" in res_rec["section"]["doc"]
    assert res_rec["section"]["other"] == 1


def test_block_scalar_anchors_tags_and_quoted_keys_preserve_hashes() -> None:
    yml = (
        'anchored: &anc | # anchor header comment\n'
        '  # Heading under anchor\n'
        'ref: *anc\n'
        'tagged: !!str | # tag header comment\n'
        '  # Heading under tag\n'
        'both: &b !!str >- # both comment\n'
        '  # Heading under both\n'
        '"a:b": | # quoted double colon comment\n'
        '  # Heading under double-quoted key\n'
        "'a:b_single': |- # quoted single colon comment\n"
        '  # Heading under single-quoted key\n'
        '"a#b": | # quoted hash key\n'
        '  # Heading under hash key\n'
        'seq:\n'
        '  - &sanc | # seq anchor comment\n'
        '    # Heading in seq anchor\n'
        'after: 42 # trailing comment\n'
    )
    stripped = _strip_yaml_comments_outside_blocks(yml)
    assert "# anchor header comment" not in stripped
    assert "# tag header comment" not in stripped
    assert "# both comment" not in stripped
    assert "# trailing comment" not in stripped
    for expected in (
        "# Heading under anchor",
        "# Heading under tag",
        "# Heading under both",
        "# Heading under double-quoted key",
        "# Heading under single-quoted key",
        "# Heading under hash key",
        "# Heading in seq anchor",
    ):
        assert expected in stripped

    res_raw = raw_yaml(yml, {})
    assert res_raw["anchored"] == "# Heading under anchor\n"
    assert res_raw["ref"] == "# Heading under anchor\n"
    assert res_raw["tagged"] == "# Heading under tag\n"
    assert res_raw["both"] == "# Heading under both"
    assert res_raw["a:b"] == "# Heading under double-quoted key\n"
    assert res_raw["a:b_single"] == "# Heading under single-quoted key"
    assert res_raw["a#b"] == "# Heading under hash key\n"
    assert res_raw["seq"] == ["# Heading in seq anchor\n"]
    assert res_raw["after"] == 42

    res_rec = recursive_yaml(yml)
    assert res_rec["a:b"] == "# Heading under double-quoted key\n"
    assert res_rec["both"] == "# Heading under both"
    assert res_rec["after"] == 42


def test_non_block_scalar_lines_not_falsely_detected() -> None:
    yml = (
        'url: http://example.com\n'
        '  # real comment line after url\n'
        'str_with_pipe: "a | b"\n'
        '  # real comment line after pipe string\n'
        'plain_pipe: foo | # inline comment\n'
        '  # real comment after plain pipe\n'
    )
    stripped = _strip_yaml_comments_outside_blocks(yml)
    assert "# real comment line after url" not in stripped
    assert "# real comment line after pipe string" not in stripped
    assert "# inline comment" not in stripped
    assert "# real comment after plain pipe" not in stripped

    res = raw_yaml(yml, {})
    assert res["url"] == "http://example.com"
    assert res["str_with_pipe"] == "a | b"
    assert res["plain_pipe"] == "foo |"


def test_ref_segments_has_unresolved_jinja_negative_index() -> None:
    assert _ref_segments_has_unresolved_jinja({"items": [10, 20, 30]}, ("items", -1), frozenset()) is False
    assert (
        _ref_segments_has_unresolved_jinja({"items": [10, 20, "{{ x }}"]}, ("items", -1), frozenset())
        is True
    )
    assert (
        _ref_segments_has_unresolved_jinja(
            {"items": [10, 20, "{{ literal }}"]},
            ("items", -1),
            frozenset({("items", 2)}),
        )
        is False
    )
    assert (
        _ref_segments_has_unresolved_jinja(
            {"items": [[1, 2], ["{{ x }}"]]},
            ("items", -1, 0),
            frozenset(),
        )
        is True
    )
    assert (
        _ref_segments_has_unresolved_jinja(
            {"items": [[1, 2], ["{{ literal }}"]]},
            ("items", -1, 0),
            frozenset({("items", 1, 0)}),
        )
        is False
    )


# 13. Iteration 4 regressions (challenger_5 findings)
def test_block_scalar_properties_before_mapping_key() -> None:
    assert (
        _strip_yaml_comments_outside_blocks(
            "&a !t doc: |-\n  # custom tag with anchor before key\nother: 1 # cmt\n"
        )
        == "&a !t doc: |-\n  # custom tag with anchor before key\nother: 1 \n"
    )

    cases: list[tuple[str, object]] = [
        (
            "- &anc doc: |\n    # seq map anchor before key\n  other: 1 # cmt\n",
            [{"doc": "# seq map anchor before key\n", "other": 1}],
        ),
        (
            "- !!str doc: |\n    # seq map tag before key\n  other: 1 # cmt\n",
            [{"doc": "# seq map tag before key\n", "other": 1}],
        ),
        (
            "&kanc doc: |\n  # anchor on key\nother: 1 # cmt\n",
            {"doc": "# anchor on key\n", "other": 1},
        ),
        (
            "!!str doc: |\n  # tag on key\nother: 1 # cmt\n",
            {"doc": "# tag on key\n", "other": 1},
        ),
        (
            "&a !!str doc: |-\n  # two props before key\nother: 1 # cmt\n",
            {"doc": "# two props before key", "other": 1},
        ),
    ]
    for yml, expected in cases:
        stripped = _strip_yaml_comments_outside_blocks(yml)
        assert "# cmt" not in stripped
        import yaml

        assert yaml.safe_load(stripped) == expected
        if isinstance(expected, dict):
            assert raw_yaml(yml, {}) == expected


def test_nextline_block_scalars_after_keys_and_dashes() -> None:
    import yaml

    cases: list[tuple[str, object]] = [
        (
            "doc:\n  |\n  # nextline after key\nother: 1 # cmt\n",
            {"doc": "# nextline after key\n", "other": 1},
        ),
        (
            "doc:\n  # comment between key and pipe\n\n  |\n  # after comment and blank\nother: 1 # cmt\n",
            {"doc": "# after comment and blank\n", "other": 1},
        ),
        (
            "doc: &anc\n  |\n  # nextline after anchor\nother: 1 # cmt\n",
            {"doc": "# nextline after anchor\n", "other": 1},
        ),
        (
            "doc:\n  &anc\n  |\n  # nextline after prop-only line\nother: 1 # cmt\n",
            {"doc": "# nextline after prop-only line\n", "other": 1},
        ),
        (
            "section:\n  doc:\n    |\n    # nextline indented\n  other: 1 # cmt\n",
            {"section": {"doc": "# nextline indented\n", "other": 1}},
        ),
        (
            "section:\n  doc:\n    |2\n    # explicit indent nextline\n   # intermediate cmt {{ bad\n  other: 1 # cmt\n",
            {"section": {"doc": "# explicit indent nextline\n", "other": 1}},
        ),
        (
            "items:\n  -\n    |\n    # nextline after dash\n  - next # cmt\n",
            {"items": ["# nextline after dash\n", "next"]},
        ),
        (
            "items:\n  - &anc\n    |\n    # nextline after dash anchor\n  - next # cmt\n",
            {"items": ["# nextline after dash anchor\n", "next"]},
        ),
        (
            "items:\n  -\n    &anc |\n    # nextline after dash prop on pipe\n  - next # cmt\n",
            {"items": ["# nextline after dash prop on pipe\n", "next"]},
        ),
        (
            "- &anc doc:\n    |\n    # seq map nextline anchor before key\n  other: 1 # cmt\n",
            [{"doc": "# seq map nextline anchor before key\n", "other": 1}],
        ),
        (
            "--- |\n  # root doc marker block\n",
            "# root doc marker block\n",
        ),
    ]
    for yml, expected in cases:
        stripped = _strip_yaml_comments_outside_blocks(yml)
        assert "# cmt" not in stripped
        assert "# intermediate cmt" not in stripped
        assert yaml.safe_load(stripped) == expected
        if isinstance(expected, dict):
            assert raw_yaml(yml, {}) == expected


def test_plain_multiline_continuation_and_intermediate_comments_stripped() -> None:
    import yaml

    cases = [
        (
            "text: hello\n  |\n    # comment in plain multiline {{ bad\nother: 1 # cmt\n",
            {"text": "hello |", "other": 1},
        ),
        (
            "text:\n  hello\n  |\n    # comment after empty key plain multiline {{ bad\nother: 1 # cmt\n",
            {"text": "hello |", "other": 1},
        ),
        (
            "items:\n  - hello\n    |\n      # comment in seq plain multiline {{ bad\n  - next # cmt\n",
            {"items": ["hello |", "next"]},
        ),
        (
            'a:\n  "hello\n  |"\n  # comment after multiline quote {{ bad\nother: 1 # cmt\n',
            {"a": "hello |", "other": 1},
        ),
        (
            "section:\n  doc: |\n    # content at 4\n   # comment at 3 with {{ bad\n  other: 1 # cmt\n",
            {"section": {"doc": "# content at 4\n", "other": 1}},
        ),
        (
            "section:\n  doc: |2\n   # comment after empty explicit-indent block at 3 with {{ bad\n  other: 1 # cmt\n",
            {"section": {"doc": "", "other": 1}},
        ),
        (
            "section:\n  doc: |\n    \n   # comment after overindented blank line with {{ bad\n  other: 1 # cmt\n",
            {"section": {"doc": "", "other": 1}},
        ),
    ]
    for yml, expected in cases:
        stripped = _strip_yaml_comments_outside_blocks(yml)
        assert "{{ bad" not in stripped
        assert yaml.safe_load(stripped) == expected
        assert raw_yaml(yml, {}) == expected
        assert dict(recursive_yaml(yml)) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "{{ " + " * ".join(["(10 ** 999)"] * 50) + " }}",
        "{{ (10 ** 999) * (10 ** 999) * (10 ** 999) * (10 ** 999) * (10 ** 999) * 1 }}",
        "{{ 'a' * 10000000 }}",
        "{{ 10000000 * 'a' }}",
        "{{ ('a' * 10000) * 1000 }}",
        "{{ [1] * 1000000 }}",
        "{{ 1000000 * [1] }}",
        "{{ ([1] * 500) * 500 }}",
        "{{ (1, 2) * 60000 }}",
        "{{ ('a' + 'b') * 60000 }}",
        "{% if false %}{{ 'a' * 1000000 }}{% endif %}",
    ],
)
def test_mul_dos_blocked_statically_across_all_entrypoints(expr: str, tmp_path: Path) -> None:
    with pytest.raises((SecurityError, YamgamSecurityError)):
        validate_expressions({"x": expr})
    with pytest.raises((SecurityError, YamgamSecurityError)):
        resolve_context({"x": expr})
    with pytest.raises((SecurityError, YamgamSecurityError)):
        render_string(expr)
    with pytest.raises((SecurityError, YamgamSecurityError)):
        render_template_text(expr)
    yaml_line = f'x: "{expr}"\n' if "'" in expr else f"x: '{expr}'\n"
    with pytest.raises((SecurityError, YamgamSecurityError)):
        raw_yaml(yaml_line, {})

    docx_in = tmp_path / "mul.docx"
    docx_out = tmp_path / "mul_out.docx"
    doc = DocxDocument()
    doc.add_paragraph(expr)
    doc.save(docx_in)
    with pytest.raises(SecurityError):
        docx_file(str(docx_in), str(docx_out), {})


def test_mul_dos_blocked_at_runtime_and_valid_multiplications_work() -> None:
    runtime_vectors: list[tuple[str, dict[str, object]]] = [
        ("{{ a * b }}", {"a": 10**2500, "b": 10**2500}),
        ("{{ a * 1 }}", {"a": 10**5000}),
        ("{{ s * n }}", {"s": "abc", "n": 40000}),
        ("{{ n * s }}", {"s": "abc", "n": 40000}),
        ("{{ s * n }}", {"s": b"xy", "n": 60000}),
        ("{{ n * s }}", {"s": bytearray(b"xy"), "n": 60000}),
        ("{{ s * n }}", {"s": [1, 2], "n": 60000}),
        ("{{ n * s }}", {"s": (1, 2), "n": 60000}),
    ]
    for expr, ctx in runtime_vectors:
        with pytest.raises((SecurityError, YamgamSecurityError)):
            render_string(expr, ctx)
        with pytest.raises((SecurityError, YamgamSecurityError)):
            render_template_text(expr, ctx)
        with pytest.raises((SecurityError, YamgamSecurityError)):
            resolve_context({**ctx, "out": expr})

    # Normal valid multiplications remain unaffected
    assert render_string("{{ 49.99 * 2 }}") == pytest.approx(99.98)
    assert render_string("{{ 10 * 20 }}") == 200
    assert render_string("{{ '=' * 40 }}") == "=" * 40
    assert render_string("{{ [0] * 10 }}") == [0] * 10
    assert render_string("{{ 'a' * 0 }}") == ""
    assert render_string("{{ 'a' * -5 }}") == ""
    assert render_string("{{ True * 5 }}") == 5


# 14. Iteration 5 regressions (reviewer_8 & challenger_8 findings)
@pytest.mark.parametrize(
    "line",
    [
        "a: " + " " * 50000 + "x\n",
        "- " + " " * 50000 + "x\n",
        "a: &anc " + " " * 50000 + "x\n",
        "- &anc " + " " * 50000 + "x\n",
        "--- " + " " * 50000 + "x\n",
        "--- &anc " + " " * 50000 + "x\n",
    ],
)
def test_strip_yaml_comments_redos_linear_time(line: str) -> None:
    t0 = time.perf_counter()
    res = _strip_yaml_comments_outside_blocks(line)
    elapsed = time.perf_counter() - t0
    assert res == line
    assert elapsed < 0.2, f"ReDoS regression: took {elapsed:.3f}s"


def test_root_doc_marker_properties_and_indented_root_block_scalars() -> None:
    positive_cases = [
        ("--- &anc\n|\n  # root\n", "# root\n"),
        ("--- !!str\n|\n  # root\n", "# root\n"),
        ("--- &anc !!str\n|\n  # root\n", "# root\n"),
        ("---\n  |\n  # root\n", "# root\n"),
        ("---\n  |2\n  # root\n # inter {{ bad\n", "# root\n"),
        ("---\n&anc\n|\n  # root\n", "# root\n"),
        ("---\n  &anc\n  |\n  # root\n", "# root\n"),
        ("--- &anc\n  |\n  # root\n", "# root\n"),
    ]
    for yml, expected in positive_cases:
        stripped = _strip_yaml_comments_outside_blocks(yml)
        assert "# root" in stripped
        assert "{{ bad" not in stripped
        assert yaml.safe_load(stripped) == expected

    negative_cases = [
        ("---\nhello\n  |\n  # cmt {{ bad\n", "hello |"),
        ("  ---\n  |\n  # cmt {{ bad\n", "--- |"),
    ]
    for yml, expected in negative_cases:
        stripped = _strip_yaml_comments_outside_blocks(yml)
        assert "{{ bad" not in stripped
        assert yaml.safe_load(stripped) == expected


def test_same_indent_nextline_block_scalars_for_indented_keys_and_dashes() -> None:
    cases: list[tuple[str, object]] = [
        (
            "section:\n  doc:\n  |\n    # hello\n  other: 1 # cmt\n",
            {"section": {"doc": "# hello\n", "other": 1}},
        ),
        (
            "section:\n  doc:\n  |2\n    # hello\n   # inter {{ bad\n  other: 1 # cmt\n",
            {"section": {"doc": "# hello\n", "other": 1}},
        ),
        (
            "items:\n  -\n  |\n    # hello\n  - next # cmt\n",
            {"items": ["# hello\n", "next"]},
        ),
        (
            "items:\n  - doc:\n    |\n      # hello\n    other: 1 # cmt\n",
            {"items": [{"doc": "# hello\n", "other": 1}]},
        ),
    ]
    for yml, expected in cases:
        stripped = _strip_yaml_comments_outside_blocks(yml)
        assert "# cmt" not in stripped
        assert "{{ bad" not in stripped
        assert "# hello" in stripped
        assert yaml.safe_load(stripped) == expected
        if isinstance(expected, dict):
            assert raw_yaml(yml, {}) == expected
            assert dict(recursive_yaml(yml)) == expected

    # Negative controls: plain scalars / sibling mappings clear pending_owner_indent
    neg_invalid = _strip_yaml_comments_outside_blocks("doc:\n  a: 1\n  |\n    # cmt {{ bad\n")
    assert "{{ bad" not in neg_invalid

    neg_plain = "doc:\nother: 1\n  |\n    # cmt {{ bad\n"
    neg_plain_stripped = _strip_yaml_comments_outside_blocks(neg_plain)
    assert "{{ bad" not in neg_plain_stripped
    assert yaml.safe_load(neg_plain_stripped) == {"doc": None, "other": "1 |"}
    assert raw_yaml(neg_plain, {}) == {"doc": None, "other": "1 |"}

