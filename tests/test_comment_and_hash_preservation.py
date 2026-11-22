from __future__ import annotations

import pytest
import typer

from yamgam import parse_yaml_layer, raw_yaml
from yamgam.utils import get_config, read_file


def test_utils_imports_on_python_312_and_get_config_handles_empty(tmp_path) -> None:
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("# just a comment\n", encoding="utf-8")

    assert get_config(str(empty_file)) is None

    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        'config:\n  theme: "#1B365D"\n  context:\n    - extra.yaml\n',
        encoding="utf-8",
    )
    cfg = get_config(str(empty_file), str(cfg_file))
    assert cfg == {"theme": "#1B365D", "context": ["extra.yaml"]}


def test_read_file_missing_raises_typer_exit(tmp_path) -> None:
    with pytest.raises(typer.Exit) as exc_info:
        read_file(str(tmp_path / "does_not_exist.yaml"))
    assert exc_info.value.exit_code == 1


def test_hash_preserved_inside_yaml_strings_and_comments_ignored(tmp_path) -> None:
    yaml_text = """
# Top-level file comment {{ not_evaluated }}
brand:
  primary_color: "#1B365D"
  ticket: "Issue #42"
  docs_url: https://example.com/docs#section-2
  language: C#
  note: ok # inline comment that should be stripped
  block: |
    # Markdown heading inside block scalar
    Item #7 stays intact
"""
    p = tmp_path / "colors.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    raw_content = read_file(str(p))
    assert '"#1B365D"' in raw_content
    assert '"Issue #42"' in raw_content

    parsed = parse_yaml_layer(raw_content)
    assert parsed["brand"]["primary_color"] == "#1B365D"
    assert parsed["brand"]["ticket"] == "Issue #42"
    assert parsed["brand"]["docs_url"] == "https://example.com/docs#section-2"
    assert parsed["brand"]["language"] == "C#"
    assert parsed["brand"]["note"] == "ok"
    assert "# Markdown heading inside block scalar" in parsed["brand"]["block"]
    assert "Item #7 stays intact" in parsed["brand"]["block"]


def test_raw_yaml_preserves_hash_in_quoted_strings_and_jinja_comments() -> None:
    yaml_text = """
{# Jinja2 comment should not break #}
color: "#1B365D"
issue: "Issue #42"
url: https://example.com/path#fragment
"""
    rendered = raw_yaml(yaml_text, {})
    assert rendered["color"] == "#1B365D"
    assert rendered["issue"] == "Issue #42"
    assert rendered["url"] == "https://example.com/path#fragment"
