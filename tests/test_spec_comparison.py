"""Tests for the spec comparison: the flag value and the flags of the disabled variant."""

from __future__ import annotations

import argparse
import json

import core_runner

BASE = ["vllm", "serve", "m", "--port", "8000"]


class _FakeServer:
    def __init__(self, command: list[str]) -> None:
        self.command = command

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def check_running(self) -> None:
        pass

    def save_metadata(self, path) -> None:
        pass


def test_strip_spec_flags_removes_every_form():
    command = [*BASE, "--speculative-config", "{}", "-sc={}", "--spec-tokens", "3",
               "--speculative-config.method", "mtp", "--async-scheduling"]
    kept, removed = core_runner.strip_spec_flags(command)
    assert kept == [*BASE, "--async-scheduling"]
    assert removed == ["--speculative-config", "{}", "-sc={}", "--spec-tokens", "3",
                       "--speculative-config.method", "mtp"]
    assert core_runner.strip_spec_flags(BASE) == (BASE, [])


def test_speculative_config_arg_sends_json_for_a_mapping():
    value = {"method": "mtp", "num_speculative_tokens": 3}
    assert json.loads(core_runner.speculative_config_arg(value)) == value
    text = '{"method":"mtp","num_speculative_tokens":3}'
    assert core_runner.speculative_config_arg(text) == text


def test_disabled_variant_has_no_spec_flag(tmp_path, monkeypatch):
    commands: dict[str, list[str]] = {}

    def fake_server(cfg, run_dir, log_name="vllm.log", command=None):
        commands[log_name] = command
        return _FakeServer(command)

    monkeypatch.setattr(core_runner, "make_managed_server", fake_server)
    monkeypatch.setattr(core_runner, "make_client", lambda cfg: object())
    monkeypatch.setattr(core_runner, "wait_for_endpoint", lambda *args: None)
    monkeypatch.setattr(core_runner, "run_decode_speed", lambda client, lengths: {})

    cfg = {"speculative_config": {"method": "mtp", "num_speculative_tokens": 3}, "decode_lengths": [512]}
    run = core_runner.Run(cfg=cfg, args=argparse.Namespace(), run_dir=tmp_path,
                          summary={"server_mode": "managed"},
                          server=_FakeServer([*BASE, "--speculative-config", '{"method":"ngram"}']))
    core_runner.step_spec_comparison(run)

    enabled = commands["vllm_spec_enabled.log"]
    assert enabled == [*BASE, "--speculative-config", '{"method": "mtp", "num_speculative_tokens": 3}']
    assert commands["vllm_spec_disabled.log"] == BASE
    removed = run.summary["spec_comparison"]["config"]["removed_server_flags"]
    assert removed == ["--speculative-config", '{"method":"ngram"}']
