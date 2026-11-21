import re
import os
import typer
import yaml

from yamgam.state import state
from yamgam.render import raw_yaml
from datetime import date, datetime

class LiteralDumper(yaml.SafeDumper):
    def represent_scalar(self, tag, value, style=None):
        # If the string contains a newline, force the literal style '|'
        if '\n' in value:
            style = '|'
        return super().represent_scalar(tag, value, style)

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))

def read_file(path: str) -> str:
    if not os.path.exists(path):
        typer.echo(f"Error: File '{path}' not found.", err=True)
        raise typer.Exit(code=1)

    with open(path, "r") as file:
        return re.sub(r'#.*', '', file.read())

def read_files(paths: list) -> str:
    content = ""
    for path in reversed(paths):
        if path is None:
            if state.debug: print("Skipping empty entry")
            continue
        if state.debug: print(f"Reading: {path}")
        content += read_file(path) + "\n"
    return content

def get_config(*paths: *str) -> dict:
    # Render first file initially to get config
    if state.debug: print("Trying to find config from files.")

    config = None
    for path in paths:
        content = read_file(path)
        rendered = raw_yaml(content)
        if "config" in rendered:
            if config is None:
                config = rendered["config"]
            else:
                typer.echo(f"Error: Multiple config declarations found.", err=True)
                raise typer.Exit(code=1)

    return config