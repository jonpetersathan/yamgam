from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any

import typer
import yaml

from yamgam.render import raw_yaml
from yamgam.state import state


class LiteralDumper(yaml.SafeDumper):
    def represent_scalar(self, tag, value, style=None):
        # If the string contains a newline, force the literal style '|'
        if '\n' in value:
            style = '|'
        return super().represent_scalar(tag, value, style)


def json_serial(obj: Any) -> str:
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def read_file(path: str) -> str:
    if not os.path.exists(path):
        typer.echo(f"Error: File '{path}' not found.", err=True)
        raise typer.Exit(code=1)

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def read_files(paths: list[str | None]) -> str:
    content = ""
    for path in reversed(paths):
        if path is None:
            if state.debug:
                print("Skipping empty entry")
            continue
        if state.debug:
            print(f"Reading: {path}")
        content += read_file(path) + "\n"
    return content


def get_config(*paths: str) -> dict[str, Any] | None:
    # Render first file initially to get config
    if state.debug:
        print("Trying to find config from files.")

    config = None
    for path in paths:
        content = read_file(path)
        rendered = raw_yaml(content)
        if isinstance(rendered, dict) and "config" in rendered:
            if config is None:
                config = rendered["config"]
            else:
                typer.echo("Error: Multiple config declarations found.", err=True)
                raise typer.Exit(code=1)

    return config