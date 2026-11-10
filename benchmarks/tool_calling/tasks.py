"""Task file loading for the tool-calling benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from harness.paths import ROOT


def resolve_tasks_path(path: str | Path) -> Path:
    """Return *path*, or the repo-root path when a relative path is missing in the cwd."""
    path = Path(path)
    if not path.is_absolute() and not path.exists() and (ROOT / path).exists():
        return ROOT / path
    return path


def load_tasks(path: str | Path) -> tuple[Any, list[dict[str, Any]]]:
    """Load the task YAML and return (config, tasks) tuple."""
    with open(path) as f:
        data = yaml.safe_load(f)

    config = data.get("version", "1.0")
    tasks = data.get("tasks", [])
    if not tasks:
        raise ValueError(f"No tasks found in {path}")
    return config, tasks
