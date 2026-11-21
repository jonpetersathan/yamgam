# Installing yamgam

This CLI tool is now fully pip-installable and can be used from anywhere in your system.

## Quick Start

### Option 1: Development Installation (Recommended for development)

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Now use yamgam from anywhere in this environment
yamgam --help
```

### Option 2: Regular Installation

```bash
pip install .
yamgam --help
```

### Option 3: Install to PATH (System-wide or user)

```bash
# For current user only
pip install --user .

# Then the yamgam command will be available in your PATH
yamgam --version
```

## Usage

Once installed, you can use yamgam directly:

```bash
# Show help
yamgam --help

# Check version
yamgam --version

# Process templates with context files
yamgam context.yaml --in template.docx --out output.docx

# Print processed context
yamgam context.yaml --print json

# Dry run (don't write changes)
yamgam context.yaml --in template.xlsx --dry-run

# Enable debug output
yamgam context.yaml --debug
```

## What Changed

- Created `setup.py` with entry points configuration
- Created `src/__init__.py` to make src a proper Python package
- Updated `src/yamgam.py` to remove the `if __name__ == "__main__"` block (unnecessary with entry points)

The `yamgam` command is now available as a console script entry point that calls the typer app directly.
