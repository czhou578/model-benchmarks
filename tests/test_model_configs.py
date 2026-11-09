"""Static checks of every model config in models/ and models/legacy/."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted([*ROOT.glob("models/*.yml"), *ROOT.glob("models/legacy/*.yml")])
# docs/results/README.md: <model>-<size>-<provider>-<quant>[-<variant>], lowercase.
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.]*(-[a-z0-9][a-z0-9.]*){3,}$")
VLLM_DEFAULT_PORT = 8000


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_configs_exist() -> None:
    assert CONFIGS


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(ROOT)))
def test_required_keys_and_name(path: Path) -> None:
    cfg = _load(path)
    assert isinstance(cfg.get("name"), str), "missing name"
    assert NAME_PATTERN.match(cfg["name"]), f"name {cfg['name']!r} breaks the naming scheme"
    assert cfg.get("endpoint", {}).get("base_url"), "missing endpoint.base_url"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(ROOT)))
def test_server_port_matches_base_url(path: Path) -> None:
    cfg = _load(path)
    command = [str(arg) for arg in cfg.get("server", {}).get("command", [])]
    if not command:
        pytest.skip("no server.command")
    port = VLLM_DEFAULT_PORT
    if "--port" in command:
        port = int(command[command.index("--port") + 1])
    assert urlparse(cfg["endpoint"]["base_url"]).port == port


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(ROOT)))
def test_speculative_config_is_json(path: Path) -> None:
    spec = _load(path).get("speculative_config")
    if spec is None:
        pytest.skip("no speculative_config")
    assert isinstance(json.loads(spec), dict)
