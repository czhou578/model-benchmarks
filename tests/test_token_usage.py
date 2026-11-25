"""Tests of benchmarks/token_usage.py with transcripts in a temporary folder and a local HTTP server."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from benchmarks import token_usage

END = 1_800_000_000   # the change time of each transcript, in s


def write_run(point, name, turns, **top):
    """A transcript as agent-eval writes it. Each turn is (prompt_tokens, completion_tokens, seconds, cached)."""
    body = [{"content": "secret text", "seconds": s,
             "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
                       "prompt_tokens_details": {"cached_tokens": cached} if cached else None} if p else {}}
            for p, c, s, cached in turns]
    path = point / name / "transcript.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task": "C1", "model": "m", "seed": 0, "turns": body} | top))
    os.utime(path, (END, END))
    return path


def test_point_events(tmp_path):
    point = tmp_path / "G1"
    write_run(point, "m-C1-s0", [(100, 10, 2.0, 0), (0, 0, 1.0, 0), (300, 30, 4.5, 64)], seconds=10.0)
    (point / "m-C2-s0").mkdir()   # a run in progress has no transcript
    runs = token_usage.point_events(point, project="bench", models={"m": "served"})
    first, last = runs.pop("m-C1-s0")   # the turn without usage (a server error) gives no event
    assert not runs
    assert first == {"source_id": "agent-eval:G1/m-C1-s0:0", "timestamp": "2027-01-15T07:59:52+00:00",
                     "thread_id": "G1/m-C1-s0", "project": "bench", "provider": "agent-eval", "model": "served",
                     "prompt_tokens": 100, "completion_tokens": 10, "cache_read_tokens": 0, "total_tokens": 110,
                     "latency_ms": 2000}
    assert last["source_id"] == "agent-eval:G1/m-C1-s0:2" and last["timestamp"] == "2027-01-15T07:59:57.500000+00:00"
    assert (last["prompt_tokens"], last["cache_read_tokens"], last["total_tokens"]) == (236, 64, 330)
    assert "secret" not in json.dumps([first, last])
    again = token_usage.point_events(point, point="run-7", skip=frozenset({"other"}))["m-C1-s0"]
    assert again[0]["thread_id"] == "run-7/m-C1-s0" and again[0]["model"] == "m" and again[0]["project"] == "agent-eval"


def test_pusher(tmp_path, monkeypatch, capsys):
    sent: list[list[dict]] = []
    monkeypatch.setattr(token_usage, "find_token", lambda: "tok")
    monkeypatch.setattr(token_usage, "post", lambda events, token: sent.append(events) or len(events))
    write_run(tmp_path, "m-C1-s0", [(100, 10, 2.0, 0), (200, 20, 2.0, 0)])
    pusher = token_usage.Pusher(tmp_path, point="p")
    assert pusher.push() == 2 and len(sent[0]) == 2
    write_run(tmp_path, "m-C2-s0", [(100, 10, 2.0, 0)])
    broken = tmp_path / "m-C3-s0" / "transcript.json"
    broken.parent.mkdir()
    broken.write_text('{"turns": [')   # agent-eval writes this file at this moment
    assert pusher.push() == 1 and sent[1][0]["thread_id"] == "p/m-C2-s0"   # only the new run
    write_run(tmp_path, "m-C3-s0", [(100, 10, 2.0, 0)])

    def down(events, token):
        raise OSError("connection refused")

    monkeypatch.setattr(token_usage, "post", down)
    assert pusher.push() == 0 and pusher.push() == 0   # no exception, and one warning
    assert capsys.readouterr().out.count("warning") == 1
    monkeypatch.setattr(token_usage, "post", lambda events, token: sent.append(events) or len(events))
    assert pusher.push() == 1 and sent[2][0]["thread_id"] == "p/m-C3-s0" and pusher.events == 4

    monkeypatch.setattr(token_usage, "find_token", lambda: None)
    assert token_usage.Pusher(tmp_path).push() == 0 and len(sent) == 3   # no token: no push


def test_find_token(tmp_path, monkeypatch):
    monkeypatch.delenv("TOKEN_TRACKER_INGEST_TOKEN", raising=False)
    monkeypatch.setattr(token_usage, "TOKEN_FILE", tmp_path / "ingest_token")
    assert token_usage.find_token() is None
    (tmp_path / "ingest_token").write_text("from-file\n")
    assert token_usage.find_token() == "from-file"
    monkeypatch.setenv("TOKEN_TRACKER_INGEST_TOKEN", "from-env")
    assert token_usage.find_token() == "from-env"


def test_post_and_main(tmp_path, monkeypatch, capsys):
    requests: list[tuple[str, str | None, dict]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            batch = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers["Authorization"], batch))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"accepted": len(batch["events"]), "received": 0}).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(token_usage, "BATCH", 2)
    monkeypatch.setattr(token_usage, "find_token", lambda: "tok")
    write_run(tmp_path, "m-C1-s0", [(100, 10, 2.0, 0)] * 3)
    argv = ["token_usage", str(tmp_path), "--url", f"http://127.0.0.1:{server.server_port}/", "--host", "box"]
    try:
        monkeypatch.setattr("sys.argv", [*argv, "--dry-run"])
        token_usage.main()
        assert not requests and "1 runs, 3 events" in capsys.readouterr().out
        monkeypatch.setattr("sys.argv", argv)
        token_usage.main()
    finally:
        server.shutdown()
    assert [(path, auth, batch["host"], len(batch["events"])) for path, auth, batch in requests] == [
        ("/api/events", "Bearer tok", "box", 2), ("/api/events", "Bearer tok", "box", 1)]
    assert "added or changed 3 rows" in capsys.readouterr().out
    monkeypatch.setattr(token_usage, "find_token", lambda: None)
    with pytest.raises(SystemExit, match="no ingest token"):
        token_usage.main()
