"""Tests for the dashboard HTTP server."""

from __future__ import annotations

import threading
import urllib.error
import urllib.request
from pathlib import Path

import serve_dashboard


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def test_root_redirects_and_page_is_rebuilt(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    out = tmp_path / "reports" / "dashboard.html"
    server = serve_dashboard.make_server("127.0.0.1", 0, results, out)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            opener.open(base + "/")
            raise AssertionError("expected a redirect from /")
        except urllib.error.HTTPError as err:
            assert err.code == 302
            assert err.headers["Location"] == "/dashboard.html"

        assert not out.exists()
        with urllib.request.urlopen(base + "/dashboard.html") as resp:
            body = resp.read().decode("utf-8")
            assert resp.status == 200
            assert resp.headers["Cache-Control"] == "no-store"
        assert out.exists()
        assert "window.BENCH" in body
    finally:
        server.shutdown()
        server.server_close()
