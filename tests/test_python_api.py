from __future__ import annotations

import copy
from datetime import date, datetime
import json

import pytest

from yamgam import (
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


def test_parse_yaml_layer_handles_empty_dates_and_errors() -> None:
    assert parse_yaml_layer("") == {}
    assert parse_yaml_layer("   \n# only comments\n") == {}

    parsed = parse_yaml_layer(
        "release: 2025-03-01\ncolor: '#1B365D'\ntitle: 'Issue #42'\n"
    )
    assert parsed == {
        "release": "2025-03-01",
        "color": "#1B365D",
        "title": "Issue #42",
    }
    json.dumps(parsed)

    with pytest.raises(YamgamYamlError):
        parse_yaml_layer("- just\n- a\n- list\n")

    with pytest.raises(YamgamYamlError) as exc_info:
        parse_yaml_layer("a: [unclosed\n")
    assert exc_info.value.line >= 1


def test_normalize_dates_recursively_json_safe() -> None:
    payload = {
        "d": date(2025, 3, 1),
        "dt": datetime(2025, 3, 1, 14, 30, 0),
        "nested": [{"at": date(2026, 1, 2)}, (datetime(2024, 12, 31, 23, 59), "ok")],
    }
    normalized = normalize_dates(payload)
    assert normalized["d"] == "2025-03-01"
    assert normalized["dt"] == "2025-03-01"
    assert normalized["nested"][0]["at"] == "2026-01-02"
    assert normalized["nested"][1] == ("2024-12-31", "ok")
    assert json.loads(json.dumps(normalized))["d"] == "2025-03-01"
    # Input untouched
    assert isinstance(payload["d"], date)
    assert isinstance(payload["dt"], datetime)


def test_non_mutating_priority_deep_merge_and_collisions() -> None:
    layer0 = {
        "company": {"name": "Base Co", "address": {"city": "NYC", "zip": "10001"}},
        "tags": ["a", "b"],
        "section": {"nested": 1},
    }
    layer1 = (
        "company:\n"
        "  address:\n"
        "    zip: '94107'\n"
        "tags:\n"
        "  - override\n"
        "section: scalar_override\n"
    )
    layer2 = {"company": {"address": {"zip": "90210"}}}

    snap0 = copy.deepcopy(layer0)
    snap2 = copy.deepcopy(layer2)

    merged = deep_merge(layer0, {"company": {"address": {"country": "US"}}})
    assert merged["company"]["address"] == {"city": "NYC", "zip": "10001", "country": "US"}
    assert layer0 == snap0

    detail = merge_layers_detailed([layer0, layer1, layer2], highest_first=False)
    assert detail.merged == {
        "company": {"name": "Base Co", "address": {"city": "NYC", "zip": "90210"}},
        "tags": ["override"],
        "section": "scalar_override",
    }
    # Verify non-mutation even after mutating result
    detail.merged["company"]["name"] = "MUTATED"
    assert layer0 == snap0
    assert layer2 == snap2

    collision_paths = {c.path: (c.overridden_index, c.winning_index) for c in detail.collisions}
    assert collision_paths["tags"] == (0, 1)
    assert collision_paths["section"] == (0, 1)
    assert collision_paths["company.address.zip"] == (1, 2)

    # highest_first=True reverses precedence
    high_first = merge_layers(
        [{"a": 1, "b": {"x": 10}}, {"a": 2, "b": {"x": 20, "y": 30}}],
        highest_first=True,
    )
    assert high_first == {"a": 1, "b": {"x": 10, "y": 30}}


def test_resolve_context_multi_hop_types_raw_preservation_and_dict_keys() -> None:
    layer0 = {
        "a": "{{ b * 4 }}",
        "b": "{{ c + 3 }}",
        "c": 2,
        "msg": "Result={{ a }}",
        "flag": True,
        "is_ok": "{{ flag }}",
        "staff": {"Barb": {"name": "Barb Dwyer"}},
        "lead": "{{ staff.Barb }}",
        "lead_name": "{{ lead.name }}",
        "order": {"items": [{"name": "Pen"}], "keys": "k1", "values": "v1", "get": "g1"},
        "first_item": "{{ order.items[0].name }}",
        "clients": {"404Dogs": {"name": "404 Small Dogs"}},
        "dog_name": "{{ clients['404Dogs'].name }}",
        "release": date(2025, 3, 1),
    }
    layer0_orig = copy.deepcopy(layer0)

    res = resolve_context([layer0])
    assert res.converged is True
    assert res.depth_used == 3
    assert res.resolved["c"] == 2
    assert res.resolved["b"] == 5
    assert res.resolved["a"] == 20
    assert res.resolved["msg"] == "Result=20"
    assert res.resolved["is_ok"] is True
    assert res.resolved["lead"] == {"name": "Barb Dwyer"}
    assert res.resolved["lead_name"] == "Barb Dwyer"
    assert res.resolved["first_item"] == "Pen"
    assert res.resolved["dog_name"] == "404 Small Dogs"
    assert res.resolved["release"] == "2025-03-01"
    json.dumps(res.resolved)

    # Raw templates preserved both in caller dict and ResolutionResult.raw
    assert layer0 == layer0_orig
    assert res.raw["a"] == "{{ b * 4 }}"
    assert res.raw["msg"] == "Result={{ a }}"

    # Convenience wrapper
    assert resolve_layers([layer0]) == res.resolved


def test_resolve_context_cycles_max_depth_and_undefined_modes() -> None:
    # Direct & mutual cycles
    with pytest.raises(YamgamMaxDepthError):
        resolve_context({"x": "{{ x }}!"})

    with pytest.raises(YamgamMaxDepthError):
        resolve_context({"a": "{{ b }}", "b": "{{ a }}"})

    non_raising = resolve_context({"a": "{{ b }}", "b": "{{ a }}"}, raise_on_max_depth=False)
    assert non_raising.converged is False
    assert set(non_raising.unresolved_paths) == {"a", "b"}

    # Chain exceeding max_depth
    long_chain = {"d": 1, "c": "{{ d + 1 }}", "b": "{{ c + 1 }}", "a": "{{ b + 1 }}"}
    with pytest.raises(YamgamMaxDepthError):
        resolve_context(long_chain, max_depth=2)
    assert resolve_layers(long_chain, max_depth=3)["a"] == 4

    # Delayed vs genuinely missing variable
    missing_ctx = {"x": "{{ missing.path }}", "y": "{{ z }}", "z": "ok", "hello": "Hi {{ missing.name }}"}
    with pytest.raises(YamgamExpressionError):
        resolve_context(missing_ctx, strict_undefined=True)

    lenient = resolve_context(missing_ctx, strict_undefined=False)
    assert lenient.converged is False
    assert lenient.resolved["y"] == "ok"
    assert lenient.resolved["hello"] == "Hi "
    assert "UNDEFINED:" not in str(lenient.resolved)
    assert "x" in lenient.errors
    assert "hello" in lenient.errors
    assert set(lenient.unresolved_paths) == {"x", "hello"}


def test_resolve_context_yaml_layers_with_comments_and_hashes() -> None:
    base_yaml = """
# Comment with {{ invalid jinja syntax that PyYAML strips cleanly
theme:
  primary: "#1B365D"
  url: https://example.org/doc#top
"""
    overlay_yaml = """
summary: "Theme {{ theme.primary }} ({{ theme.url }})"
"""
    res = resolve_context([base_yaml, overlay_yaml])
    assert res.converged is True
    assert res.resolved["theme"]["primary"] == "#1B365D"
    assert res.resolved["summary"] == "Theme #1B365D (https://example.org/doc#top)"
    assert res.raw["summary"] == "Theme {{ theme.primary }} ({{ theme.url }})"


def test_extract_validate_and_pure_rendering_helpers() -> None:
    payload = {
        "title": 'Hi {{ "a }} b" }} {{ user.name }}',
        "total": "{% for row in rows %}{{ row.price + tax }}{% endfor %}",
    }
    exprs = extract_expressions(payload)
    assert exprs == [
        '{{ "a }} b" }}',
        "{{ user.name }}",
        "{% for row in rows %}",
        "{{ row.price + tax }}",
        "{% endfor %}",
    ]

    refs = extract_referenced_variables(payload)
    assert refs == ["user.name", "rows", "tax"]

    assert validate_expressions(payload) == []

    with pytest.raises(YamgamSecurityError):
        validate_expressions({"x": "{% if false %}{{ ''.__class__ }}{% endif %}"})

    with pytest.raises(YamgamSecurityError):
        validate_expressions({"x": "{% if ok %}yes{% endif %}"}, allow_statements=False)

    errs = validate_expressions(
        {"bad": "{{ broken", "evil": "{{ ''.__class__ }}"},
        raise_on_error=False,
    )
    assert len(errs) == 2
    assert {e["error_type"] for e in errs} == {"syntax", "security"}

    assert render_string("{{ count + 1 }}", {"count": 4}) == 5
    assert render_template_text("{{ count + 1 }}", {"count": 4}) == "5"
    assert render_template_text("Hello {{ who }}", {"who": "DocTag"}) == "Hello DocTag"
