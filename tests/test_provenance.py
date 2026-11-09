"""Tests for the run provenance and the server record. No server, no GPU."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import requests

import harness.provenance as provenance
from vllm_server import VllmServer, is_secret_name


def _fake_git(outputs: dict[str, tuple[int, str]]):
    def run(cmd, **kwargs):
        code, out = outputs[cmd[1]]
        return subprocess.CompletedProcess(cmd, code, stdout=out, stderr="")
    return run


def test_harness_provenance_reads_commit_and_dirty_state(monkeypatch) -> None:
    monkeypatch.setattr(provenance.subprocess, "run", _fake_git({
        "rev-parse": (0, "abc123\n"), "status": (0, " M core_runner.py\n"),
    }))
    assert provenance.harness_provenance() == {
        "schema_version": provenance.SCHEMA_VERSION, "git_commit": "abc123", "git_dirty": True,
    }


def test_harness_provenance_clean_checkout(monkeypatch) -> None:
    monkeypatch.setattr(provenance.subprocess, "run", _fake_git({
        "rev-parse": (0, "abc123\n"), "status": (0, ""),
    }))
    assert provenance.harness_provenance()["git_dirty"] is False


@pytest.mark.parametrize("failure", [FileNotFoundError("git"), "not-a-repo"])
def test_harness_provenance_without_git(monkeypatch, failure) -> None:
    def run(cmd, **kwargs):
        if isinstance(failure, Exception):
            raise failure
        return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="fatal")
    monkeypatch.setattr(provenance.subprocess, "run", run)
    info = provenance.harness_provenance()
    assert info == {"schema_version": 3, "git_commit": None, "git_dirty": None}


class _Response:
    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def test_fetch_server_version_reads_the_version_endpoint(monkeypatch) -> None:
    seen = {}

    def get(url, headers=None, timeout=None):
        seen.update(url=url, headers=headers)
        return _Response(200, {"version": "0.33.0"})
    monkeypatch.setattr(provenance.requests, "get", get)
    version = provenance.fetch_server_version("http://h:8000/", {"Authorization": "Bearer x"})
    assert version == "0.33.0"
    assert seen == {"url": "http://h:8000/version", "headers": {"Authorization": "Bearer x"}}


@pytest.mark.parametrize("outcome", [
    _Response(404, {"detail": "Not Found"}),
    _Response(200, ValueError("not json")),
    _Response(200, {"version": None}),
    requests.ConnectionError("refused"),
])
def test_fetch_server_version_returns_none_on_failure(monkeypatch, outcome) -> None:
    def get(*args, **kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(provenance.requests, "get", get)
    assert provenance.fetch_server_version("http://h:8000") is None


@pytest.mark.parametrize("name", [
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY", "VLLM_API_KEY", "API_KEY",
    "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "DB_PASSWORD", "MY_SECRET_VALUE",
    "hf_token", "TOKEN", "SMTP_PASS",
])
def test_secret_names_are_redacted(name: str) -> None:
    assert is_secret_name(name)


@pytest.mark.parametrize("name", [
    "VLLM_USE_FASTOKENS", "TOKENIZERS_PARALLELISM", "VLLM_MAX_TOKENS", "CUDA_VISIBLE_DEVICES",
    "VLLM_KEY_CACHE_DTYPE", "BYPASS", "MONKEY",
])
def test_plain_names_are_not_redacted(name: str) -> None:
    assert not is_secret_name(name)


def test_metadata_redacts_secrets_and_stores_a_relative_log_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "m" / "20261110_000000"
    run_dir.mkdir(parents=True)
    server = VllmServer(
        ["vllm", "serve", "m", "--port", "8000"], "http://127.0.0.1:8000",
        log_path=run_dir / "vllm_spec_enabled.log",
        environment={"HF_TOKEN": "hf_secret", "VLLM_USE_FASTOKENS": "1"},
    )
    server.save_metadata(run_dir / "resolved_server.json")
    saved = json.loads((run_dir / "resolved_server.json").read_text())
    assert saved["environment"] == {"HF_TOKEN": "<redacted>", "VLLM_USE_FASTOKENS": "1"}
    assert saved["log_path"] == "vllm_spec_enabled.log"
    assert "hf_secret" not in json.dumps(saved)
    # The object keeps the full path for the code.
    assert server.log_path == run_dir / "vllm_spec_enabled.log"
    assert server.metadata(base_dir=tmp_path / "elsewhere")["log_path"] == "vllm_spec_enabled.log"
    assert server.metadata(base_dir=tmp_path)["log_path"] == "results/m/20261110_000000/vllm_spec_enabled.log"
