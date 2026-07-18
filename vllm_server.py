"""Small, safe lifecycle wrapper for a runner-managed vLLM server."""

from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


class VllmServerError(RuntimeError):
    """Raised when a managed vLLM server cannot be started or kept alive."""


class VllmServer:
    def __init__(
        self,
        command: Sequence[str],
        base_url: str,
        log_path: Path,
        environment: Mapping[str, str] | None = None,
        shutdown_timeout_s: float = 30,
    ) -> None:
        if not command or not all(isinstance(arg, str) and arg for arg in command):
            raise ValueError("server.command must be a non-empty list of strings")

        parsed = urlparse(base_url)
        if not parsed.hostname or not parsed.port:
            raise ValueError(f"endpoint base_url must include a host and port: {base_url}")

        self.command = list(command)
        self.base_url = base_url.rstrip("/")
        self.host = parsed.hostname
        self.port = parsed.port
        self.log_path = Path(log_path)
        self.environment = {str(k): str(v) for k, v in (environment or {}).items()}
        self.shutdown_timeout_s = shutdown_timeout_s
        self.process: subprocess.Popen[str] | None = None
        self.started_at: float | None = None
        self.stopped_at: float | None = None
        self._log_file = None

        command_port = self._option_value("--port")
        if command_port is not None and int(command_port) != self.port:
            raise ValueError(
                f"server command port {command_port} does not match endpoint port {self.port}"
            )

    def _option_value(self, option: str) -> str | None:
        try:
            return self.command[self.command.index(option) + 1]
        except (ValueError, IndexError):
            return None

    def _port_is_open(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=0.25):
                return True
        except OSError:
            return False

    def start(self) -> None:
        if self.process is not None:
            raise VllmServerError("managed vLLM server has already been started")
        if self._port_is_open():
            raise VllmServerError(
                f"refusing to start vLLM: {self.host}:{self.port} is already in use; "
                "use --server-mode external to benchmark that endpoint"
            )

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env.update(self.environment)
        self.started_at = time.time()

        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=True,
            )
        except Exception:
            self._close_log()
            raise

    def check_running(self) -> None:
        if self.process is None:
            raise VllmServerError("managed vLLM server has not been started")
        return_code = self.process.poll()
        if return_code is not None:
            self._close_log()
            raise VllmServerError(
                f"vLLM exited during startup with code {return_code}\n{self.log_tail()}"
            )

    def stop(self) -> None:
        process = self.process
        if process is None:
            return

        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=self.shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)

        self.stopped_at = time.time()
        self._close_log()

    def _close_log(self) -> None:
        if self._log_file is not None and not self._log_file.closed:
            self._log_file.flush()
            self._log_file.close()

    def log_tail(self, lines: int = 30) -> str:
        if not self.log_path.exists():
            return "(vLLM log is empty)"
        content = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])

    def metadata(self) -> dict[str, Any]:
        sensitive = ("TOKEN", "KEY", "SECRET", "PASSWORD")
        environment = {
            key: "<redacted>" if any(word in key.upper() for word in sensitive) else value
            for key, value in self.environment.items()
        }
        return {
            "command": self.command,
            "command_display": shlex.join(self.command),
            "base_url": self.base_url,
            "host": self.host,
            "port": self.port,
            "environment": environment,
            "pid": self.process.pid if self.process else None,
            "started_at_unix": self.started_at,
            "stopped_at_unix": self.stopped_at,
            "return_code": self.process.poll() if self.process else None,
            "log_path": str(self.log_path),
        }

    def save_metadata(self, path: Path) -> None:
        path.write_text(json.dumps(self.metadata(), indent=2), encoding="utf-8")

    def __enter__(self) -> "VllmServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
