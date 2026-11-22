import os
import typer

from enum import Enum
from pprint import pprint

from yamgam import render
from yamgam import utils
from yamgam.state import state

app = typer.Typer(add_completion=False)

class OutputFormat(str, Enum):
    text = "text"
    yaml = "yaml"
    json = "json"

def version_callback(value: bool):
    if value:
        print(f"yamgam {state.version}")
        raise typer.Exit()

@app.command()
def main( 
    paths:          list[str]    = typer.Argument(["context.yaml"],              help="Context files to process"),
    input_file:     str          = typer.Option(None,    "--in",         "-i",   help="Path to template input file"),
    output_file:    str          = typer.Option(None,    "--out",        "-o",   help="Path to rendered output file"),
    config_file:    str          = typer.Option(None,    "--config",     "-c",   help="Path to config file"),
    print_only:     OutputFormat = typer.Option(None,    "--print",      "-p",   help="Outputs only processed context to console"),
    dry_run:        bool         = typer.Option(False,   "--dry-run",            help="Does not write any changes to disk"),
    depth:          int          = typer.Option(10,      "--depth",      "-n",   help="Maximum depth when processing context"),
    debug:          bool         = typer.Option(False,   "--debug",      "-d",   help="Prints additional debug messages when running"),
    verbose:        bool         = typer.Option(False,   "--verbose",    "-v",   help="Make the operation more talkative"),
    version:        bool | None  = typer.Option(None,    "--version",            help="Output version information and exit", callback=version_callback, is_eager=True)):

    state.debug = debug
    if debug: print("Debug mode enabled")
    state.verbose = verbose or debug
    if verbose: print(f"Running yamgam {state.version}")
    state.depth = depth
    if debug: print(f"Using maximum depth of {state.depth}")

    config = None
    if config_file:
        config = utils.get_config(config_file)
    else:
        config = utils.get_config(*paths)
    
    is_config_needed = not (print_only or (input_file and output_file) or (dry_run and input_file))
    if config is None and is_config_needed:
        typer.echo(f"Error: No config found and not running in print (-p) mode.", err=True)
        raise typer.Exit(code=1)

    generators = list()
    if input_file:
        generators.append({
            "input": input_file,
            "output": output_file
        })
    elif config and "generators" in config:
        if not isinstance(config["generators"], list):
            typer.echo(f"Error: Invalid generator config.", err=True)
            raise typer.Exit(code=1)
        for generator in config["generators"]:
            if "input" not in generator:
                typer.echo(f"Error: Invalid generator config. Input missing.", err=True)
                raise typer.Exit(code=1)
            if "output" not in generator:
                typer.echo(f"Error: Invalid generator config. Output missing.", err=True)
                raise typer.Exit(code=1)
            generators.append({
                "name": generator.get("name"),
                "input": generator["input"],
                "output": generator["output"]
            })

    if config and "context" in config:
        if debug: print("Adding additional context to file list.")
        paths = paths + config["context"]

    paths = list(p for p in paths if p is not None)
    if verbose: print(f"Context: {', '.join(paths)}")

    context_raw = utils.read_files(paths)

    # Rendering context
    if debug: print("Rendering context...")
    context = render.recursive_yaml(context_raw, depth)

    if print_only:
        if print_only == OutputFormat.text:
            pprint(context.to_dict())
        elif print_only == OutputFormat.json:
            print(context.to_json(default=utils.json_serial))
        elif print_only == OutputFormat.yaml:
            print(context.to_yaml(Dumper=utils.LiteralDumper))
        else:
            typer.echo(f"Error: Invalid print output type.", err=True)
            raise typer.Exit(code=1)
    else:
        count = 0
        for generator in generators:
            count += 1
            renderer = None
            if state.verbose:
                if "name" in generator:
                    print(f"Processing generator {generator['name']}")
                else:
                    print(f"Processing generator {count}")

            input_file = generator["input"]
            output_file = generator["output"]
            if not os.path.exists(input_file):
                typer.echo(f"Error: Input template {input_file} does not exist.", err=True)
                raise typer.Exit(code=1)
            file_type = input_file.split(".")[-1]
            if file_type == "docx":
                renderer = render.docx_file
            elif file_type == "xlsx":
                renderer = render.xlsx_file
            elif file_type == "pptx":
                renderer = render.pptx_file
            else:
                typer.echo(f"Error: Template filetype {file_type} is not supported.", err=True)
                raise typer.Exit(code=1)
            if dry_run:
                renderer(input_file, None, context)
            else:
                renderer(input_file, output_file, context)

    #TODO: Error handling
    # Search recursively through the attr for all UNDEFINED:XXX and trace the line in the yaml and print it

if __name__ == "__main__":
    app()