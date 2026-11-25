"""Send the token usage of an agent-eval point folder to Token Tracker.

agent-eval calls the model server directly, so Token Tracker does not see these tokens. This module reads each
``<point>/<run>/transcript.json`` and sends one usage event for each model turn to the ingest endpoint of Token
Tracker (``POST /api/events``). The events have the provider ``agent-eval``.

    python -m benchmarks.token_usage POINT_DIR [--model NAME=SERVED_NAME] [--project NAME] [--point NAME]

Each turn has a fixed ``source_id`` (``agent-eval:<point>/<run>:<turn>``). Token Tracker keeps one row for each
``source_id``, so a second push adds nothing. Give each point a different ``--point`` name: a second point with the
same name replaces the rows of the first point.

The counts come from the ``usage`` object of the server. ``completion_tokens`` includes the reasoning tokens.
``prompt_tokens`` is the count of the server minus the cached tokens, and the cached tokens go to
``cache_read_tokens``. ``latency_ms`` is ``seconds`` of the turn.

A transcript has no time for each turn. The module calculates the time: the end of the run is the change time of
``transcript.json``, the start is the end minus ``seconds`` of the run, and a turn ends at the start plus the
``seconds`` of the turns up to it. Thus the time between the turns (the tools) is not in the result, and the times
are wrong if a later command writes the transcript again before the first push.

The module sends counts, model names, task ids and seeds only. It sends no prompt and no transcript text.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROVIDER = "agent-eval"
DEFAULT_URL = "http://127.0.0.1:8765"
TOKEN_FILE = Path("~/.tokentracker/ingest_token")
BATCH = 500   # the endpoint accepts 1000 events in one request


def find_token() -> str | None:
    """The ingest token of Token Tracker: TOKEN_TRACKER_INGEST_TOKEN, or the token file. None: no push."""
    token = os.environ.get("TOKEN_TRACKER_INGEST_TOKEN", "").strip()
    try:
        return token or TOKEN_FILE.expanduser().read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def run_events(transcript: Path, thread: str, project: str, models: dict[str, str]) -> list[dict[str, Any]]:
    """One event for each turn of a transcript that has token counts."""
    tr = json.loads(transcript.read_text(encoding="utf-8"))
    turns = tr.get("turns") or []
    end = datetime.fromtimestamp(transcript.stat().st_mtime, tz=timezone.utc)
    at = end - timedelta(seconds=float(tr.get("seconds") or sum(float(t.get("seconds") or 0) for t in turns)))
    model = models.get(tr.get("model"), tr.get("model"))
    events = []
    for i, turn in enumerate(turns):
        usage, seconds = turn.get("usage") or {}, float(turn.get("seconds") or 0)
        at = min(at + timedelta(seconds=seconds), end)
        prompt, completion = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        cached = min(int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0), prompt)
        if prompt + completion:
            events.append({
                "source_id": f"{PROVIDER}:{thread}:{i}", "timestamp": at.isoformat(), "thread_id": thread,
                "project": project, "provider": PROVIDER, "model": model, "prompt_tokens": prompt - cached,
                "completion_tokens": completion, "cache_read_tokens": cached, "total_tokens": prompt + completion,
                "latency_ms": round(seconds * 1000),
            })
    return events


def point_events(point_dir: Path, *, point: str | None = None, project: str = PROVIDER,
                 models: dict[str, str] | None = None, skip: frozenset[str] = frozenset()) -> dict[str, list[dict]]:
    """The events of each finished run folder of a point, by folder name. A transcript that does not parse yet
    (agent-eval writes it at this moment) is left for the next call."""
    runs = {}
    for transcript in sorted(point_dir.glob("*/transcript.json")):
        name = transcript.parent.name
        if name not in skip:
            try:
                runs[name] = run_events(transcript, f"{point or point_dir.name}/{name}", project, models or {})
            except (OSError, ValueError, TypeError, AttributeError):
                continue
    return runs


def post(events: list[dict], *, token: str, url: str | None = None, host: str | None = None) -> int:
    """Send the events. Returns the number of rows that Token Tracker added or changed."""
    url = (url or os.environ.get("TOKEN_TRACKER_URL") or DEFAULT_URL).rstrip("/") + "/api/events"
    host = host or os.environ.get("TOKEN_TRACKER_HOSTNAME") or socket.gethostname()
    accepted = 0
    for i in range(0, len(events), BATCH):
        body = json.dumps({"host": host, "events": events[i:i + BATCH]}).encode()
        request = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            accepted += json.load(response)["accepted"]
    return accepted


class Pusher:
    """Sends the runs of a point that it did not send before, on each call of push(). A failure gives one warning
    and no exception, because the benchmark must continue. Without an ingest token, push() does nothing."""

    def __init__(self, point_dir: Path, **labels: Any) -> None:
        self.point_dir, self.labels, self.token = point_dir, labels, find_token()
        self.sent: set[str] = set()
        self.events, self.warned = 0, False

    def push(self) -> int:
        if not self.token:
            return 0
        try:
            runs = point_events(self.point_dir, skip=frozenset(self.sent), **self.labels)
            events = [event for run in runs.values() for event in run]
            post(events, token=self.token)
        except Exception as exc:
            if not self.warned:
                print(f"[token-usage] warning: the push to Token Tracker failed: {type(exc).__name__}: {exc}")
            self.warned = True
            return 0
        self.sent |= runs.keys()
        self.events += len(events)
        return len(events)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("point_dir", type=Path, help="a point folder of agent-eval: it has one folder for each run")
    ap.add_argument("--point", help="the name of the point in the thread id (default: the folder name)")
    ap.add_argument("--project", default=PROVIDER, help="the project name on the dashboard (default: %(default)s)")
    ap.add_argument("--model", action="append", default=[], metavar="NAME=SERVED_NAME",
                    help="show the agent-eval model NAME as SERVED_NAME; give it again for each model")
    ap.add_argument("--url", help=f"the address of Token Tracker (default: TOKEN_TRACKER_URL or {DEFAULT_URL})")
    ap.add_argument("--host", help="the machine name on the dashboard (default: the name of this machine)")
    ap.add_argument("--dry-run", action="store_true", help="print the totals and send nothing")
    a = ap.parse_args()
    runs = point_events(a.point_dir, point=a.point, project=a.project, models=dict(m.split("=", 1) for m in a.model))
    events = [event for run in runs.values() for event in run]
    totals = {key: sum(event[key] for event in events) for key in ("prompt_tokens", "completion_tokens",
                                                                   "cache_read_tokens", "total_tokens")}
    print(f"[token-usage] {len(runs)} runs, {len(events)} events, {totals}")
    if a.dry_run:
        return
    token = find_token()
    if not token:
        raise SystemExit(f"no ingest token: set TOKEN_TRACKER_INGEST_TOKEN or write {TOKEN_FILE}")
    print(f"[token-usage] Token Tracker added or changed {post(events, token=token, url=a.url, host=a.host)} rows")


if __name__ == "__main__":
    main()
