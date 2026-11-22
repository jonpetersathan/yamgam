from __future__ import annotations

from pathlib import Path

from docx import Document as DocxDocument
from jinja2.exceptions import SecurityError
import openpyxl
from pptx import Presentation
import pytest

from yamgam import (
    YamgamSecurityError,
    extract_expressions,
    extract_referenced_variables,
    merge_layers,
    parse_yaml_layer,
    raw_yaml,
    recursive_yaml,
    render_string,
    render_template_text,
    resolve_context,
    resolve_layers,
    validate_expressions,
)
from yamgam.render import docx_file, pptx_file, xlsx_file
from yamgam.utils import get_config

SSTI_PAYLOADS = [
    '{{ "".__class__.__mro__ }}',
    '{{ ().__class__.__bases__[0].__subclasses__() }}',
    '{{ cycler.__init__.__globals__ }}',
    '{{ lipsum.__globals__ }}',
    '{{ "" | attr("__class__") }}',
    '{{ [""] | map("attr", "__class__") | list }}',
    '{{ [{"a": 1}] | selectattr("__class__") | list }}',
    '{% if false %}{{ "".__class__ }}{% endif %}',
    '{% import "os" as os %}{{ os.name }}',
    '{% include "secret.txt" %}',
    '{% extends "base.html" %}',
    '{% macro evil() %}{{ "".__class__ }}{% endmacro %}',
    '{{ data["_secret"] }}',
    '{{ "{0.__class__}".format("x") }}',
    '{{ 2 ** 10000 }}',
    '{{ "" | attr("__cl" ~ "ass__") }}',
    '{{ [{"a": 1}] | map(attribute="__class__") | list }}',
]


@pytest.mark.parametrize("payload", SSTI_PAYLOADS)
def test_ssti_blocked_across_validation_and_evaluation(payload: str) -> None:
    with pytest.raises((SecurityError, YamgamSecurityError)):
        validate_expressions({"x": payload})

    with pytest.raises((SecurityError, YamgamSecurityError)):
        resolve_context([{"data": {"_secret": "nope"}, "x": payload}])

    with pytest.raises((SecurityError, YamgamSecurityError)):
        render_string(payload, {"data": {"_secret": "nope"}})

    with pytest.raises((SecurityError, YamgamSecurityError)):
        render_template_text(payload, {"data": {"_secret": "nope"}})

    with pytest.raises((SecurityError, YamgamSecurityError)):
        raw_yaml(f"x: '{payload}'", {"data": {"_secret": "nope"}})

    with pytest.raises((SecurityError, YamgamSecurityError)):
        extract_referenced_variables(payload)


def test_office_renderers_block_ssti_and_fix_pptx_signature(tmp_path: Path) -> None:
    # DOCX
    docx_in = tmp_path / "evil.docx"
    docx_out = tmp_path / "evil_out.docx"
    d = DocxDocument()
    d.add_paragraph('{{ "".__class__.__mro__ }}')
    d.save(docx_in)
    with pytest.raises(SecurityError):
        docx_file(str(docx_in), str(docx_out), {})

    # XLSX
    xlsx_in = tmp_path / "evil.xlsx"
    xlsx_out = tmp_path / "evil_out.xlsx"
    wb = openpyxl.Workbook()
    wb.active["A1"] = '{{ "".__class__.__mro__ }}'
    wb.save(xlsx_in)
    with pytest.raises(SecurityError):
        xlsx_file(str(xlsx_in), str(xlsx_out), {})

    # PPTX (also verifies _PPTXRendering 2-arg fix and happy path)
    pptx_in = tmp_path / "safe.pptx"
    pptx_out = tmp_path / "safe_out.pptx"
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    tb = slide.shapes.add_textbox(0, 0, 1000000, 500000)
    tb.text_frame.text = "Hello {{ name }}"
    prs.save(pptx_in)
    pptx_file(str(pptx_in), str(pptx_out), {"name": "World"})
    rendered_prs = Presentation(pptx_out)
    assert rendered_prs.slides[0].shapes[0].text_frame.text == "Hello World"

    pptx_evil = tmp_path / "evil.pptx"
    prs_evil = Presentation()
    slide2 = prs_evil.slides.add_slide(prs_evil.slide_layouts[6])
    tb2 = slide2.shapes.add_textbox(0, 0, 1000000, 500000)
    tb2.text_frame.text = '{{ "".__class__.__mro__ }}'
    prs_evil.save(pptx_evil)
    with pytest.raises(SecurityError):
        pptx_file(str(pptx_evil), str(tmp_path / "evil_out.pptx"), {})


def test_zero_stdout_and_stderr_leakage(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("config:\n  name: test\n", encoding="utf-8")

    _ = get_config(str(cfg))
    _ = parse_yaml_layer("a: 1\nb: '#1B365D'\n")
    _ = merge_layers([{"a": 1}, {"b": "{{ a + 2 }}"}])
    _ = raw_yaml("a: 1\nb: '{{ a + 2 }}'\n", {"a": 1})
    _ = recursive_yaml("a: 1\nb: '{{ a + 2 }}'\n", depth=5)
    _ = resolve_context([{"a": 1, "b": "{{ a + 2 }}"}])
    _ = resolve_layers([{"a": 1, "b": "{{ a + 2 }}"}])
    _ = validate_expressions({"a": "{{ x + 1 }}"})
    _ = extract_expressions("Hi {{ user.name }}")
    _ = extract_referenced_variables("Hi {{ user.name }}")
    _ = render_string("{{ a + 1 }}", {"a": 4})
    _ = render_template_text("Hi {{ name }}", {"name": "Ada"})

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
