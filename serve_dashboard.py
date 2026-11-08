#!/usr/bin/env python3
"""Rebuild the dashboard and serve it over HTTP, for one-press viewing.

Usage:
    python3 serve_dashboard.py                  # http://127.0.0.1:8080/dashboard.html
    python3 serve_dashboard.py --port 8090
    python3 serve_dashboard.py --host 0.0.0.0   # reachable from other machines

The server rebuilds reports/dashboard.html from results/ on every request
for the page, so a browser reload after a benchmark run shows the new run.
It needs only the standard library. The "Dashboard" entry in
.vscode/launch.json runs this script and opens the page in the browser.
"""

from __future__ import annotations

import argparse
import errno
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import dashboard

ROOT = Path(__file__).resolve().parent
PAGE = "dashboard.html"
DEFAULT_OUT = ROOT / "reports" / PAGE


def rebuild(results_dir: Path, out: Path) -> int:
    """Write the dashboard page and return the number of runs it holds."""
    payload = dashboard.build_payload(results_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dashboard.render(payload, fragment=False), encoding="utf-8")
    return len(payload["runs"])


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serve the reports folder. Rebuild the page on each request for it."""

    results_dir: Path = ROOT / "results"
    out: Path = DEFAULT_OUT

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        path = self.path.split("?", 1)[0]
        if path == "/":
            self.send_response(302)
            self.send_header("Location", "/" + PAGE)
            self.end_headers()
            return
        if path == "/" + PAGE:
            try:
                rebuild(self.results_dir, self.out)
            except Exception as exc:  # keep the last good page on screen
                self.log_message("rebuild failed, serving the previous page: %s", exc)
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def make_server(host: str, port: int, results_dir: Path, out: Path) -> ThreadingHTTPServer:
    """Bind a server that serves ``out.parent`` and rebuilds ``out`` from ``results_dir``."""

    class Handler(DashboardHandler):
        pass

    Handler.results_dir = results_dir
    Handler.out = out
    return ThreadingHTTPServer((host, port), partial(Handler, directory=str(out.parent)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    results_dir, out = Path(args.results), Path(args.out)

    shown_host = "127.0.0.1" if args.host in ("", "0.0.0.0") else args.host
    url = f"http://{shown_host}:{args.port}/{PAGE}"
    try:
        server = make_server(args.host, args.port, results_dir, out)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        print(f"port {args.port} is in use; a dashboard server is probably running already")
        print(f"Serving dashboard at {url}", flush=True)
        return

    runs = rebuild(results_dir, out)
    print(f"dashboard: {runs} runs -> {out}")
    print(f"Serving dashboard at {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
