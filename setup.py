from setuptools import setup, find_packages

setup(
    name="yamgam",
    version="0.1.0",
    description="A template rendering tool for docx, xlsx, and pptx files",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "docxtpl",
        "pyyaml",
        "typer",
        "python-box",
        "xlsxtpl",
        "python-pptx",
        "openpyxl",
        "jinja2",
    ],
    entry_points={
        "console_scripts": [
            "yamgam=yamgam.cli:app",
        ],
    },
)
