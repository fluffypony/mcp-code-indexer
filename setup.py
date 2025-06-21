"""Setup configuration for MCP Code Indexer."""

from setuptools import setup, find_packages
from pathlib import Path
import sys

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


# Read version from pyproject.toml
def get_version():
    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib

        with open(this_directory / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        return data["project"]["version"]
    except Exception as e:
        # Fail hard if version reading fails
        raise RuntimeError(f"Could not read version from pyproject.toml: {e}")


setup(
    name="mcp-code-indexer",
    version=get_version(),
    description="MCP server that tracks file descriptions across codebases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MCP Code Indexer Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "tiktoken>=0.9.0",
        "mcp>=1.9.0",
        "gitignore_parser==0.1.11",
        "pydantic>=2.8.0",
        "aiofiles==23.2.0",
        "aiosqlite==0.19.0",
        "aiohttp>=3.8.0",
        "tenacity>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "mcp-code-index=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="mcp model-context-protocol code-indexer file-tracking",
    project_urls={
        "Bug Reports": "https://github.com/fluffypony/mcp-code-indexer/issues",
        "Source": "https://github.com/fluffypony/mcp-code-indexer",
    },
)
