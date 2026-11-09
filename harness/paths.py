"""Repository paths."""

from __future__ import annotations

from pathlib import Path

# The repo root: the folder that holds core_runner.py.
ROOT = Path(__file__).resolve().parents[1]


def repo_relative(path: Path) -> str:
    """Return the path relative to the repo root when the path is inside it.

    The results JSON then holds no absolute home directory paths.
    """
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)

