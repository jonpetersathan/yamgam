from __future__ import annotations

from pathlib import Path

from box import Box
import pytest

from yamgam.render import raw_yaml, recursive_yaml
from yamgam.utils import get_config, read_files

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_examples_context_yaml_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(EXAMPLES_DIR)
    config = get_config("context.yaml")
    assert config is not None
    assert "context" in config

    paths = ["context.yaml"] + config["context"]
    raw_content = read_files(paths)
    result = recursive_yaml(raw_content, depth=10)

    assert isinstance(result, Box)
    assert result.sales[0].staff == "Otto Mobile"
    assert result.sales[0].product == "Left-Handed Screwdriver"
    assert result.sales[0].quantity == 14
    assert result.sales[0].revenue == pytest.approx(51.99)
    assert result.products.item1.responsible.name == "Barb Dwyer"
    assert result.clients["404Dogs"].name == "404 Small Dogs"

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_raw_yaml_and_recursive_yaml_handle_empty_and_cycles() -> None:
    assert raw_yaml("", {}) == {}
    assert raw_yaml("# comment only\n", {}) == {}
    assert recursive_yaml("", depth=5) == Box({})
    assert recursive_yaml("", depth=0) == Box({})

    # Mutual cycle must terminate cleanly at depth without hanging
    cyclic = "a: '{{ b }}'\nb: '{{ a }}'\n"
    res = recursive_yaml(cyclic, depth=3)
    assert isinstance(res, Box)
    assert "a" in res and "b" in res
