"""Provenance of a run: the harness schema version, the git commit, and the
vLLM version that the server reports."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping

import requests

from harness.paths import ROOT

# SCHEMA_VERSION N marks a run of era EN in docs/comparability.md. Runs of
# eras E0 to E2 have no version in summary.json; a provenance.json file in
# their run dir gives their era. Increase this number when a change starts a
# new era, and add the era to docs/comparability.md.
SCHEMA_VERSION = 4


def _git(args: list[str], repo: Path) -> str | None:
    """Return the stripped stdout of ``git <args>`` in *repo*, or None on failure."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def harness_provenance(repo: Path = ROOT) -> dict[str, Any]:
    """Return the schema version and the git state of the harness checkout.

    ``git_dirty`` is True when a tracked file differs from the commit.
    Untracked files, for example new run dirs, do not count. Both git
    fields are None when git or the repository is not available.
    """
    commit = _git(["rev-parse", "HEAD"], repo)
    status = _git(["status", "--porcelain", "--untracked-files=no"], repo) if commit else None
    return {
        "schema_version": SCHEMA_VERSION,
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


def fetch_server_version(base_url: str, headers: Mapping[str, str] | None = None,
                         timeout_s: float = 5.0) -> str | None:
    """Return the version from ``GET <base_url>/version``, or None on any failure.

    vLLM serves this endpoint. The server can run in another Python
    environment than the runner, so this version can differ from the
    ``vllm_version`` in environment.json.
    """
    try:
        r = requests.get(f"{base_url.rstrip('/')}/version",
                         headers=dict(headers or {}), timeout=timeout_s)
        if r.status_code != 200:
            return None
        version = r.json().get("version")
    except Exception:
        return None
    return version if isinstance(version, str) and version else None
