"""Small, safe lifecycle wrapper for a runner-managed vLLM server."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlparse

# A variable name is secret-like when it ends in one of these words, after
# "_" or at the start, or when it contains SECRET or PASSWORD. Thus HF_TOKEN
# and OPENAI_API_KEY are redacted, but VLLM_USE_FASTOKENS is not.
_SECRET_NAME = re.compile(
    r"(?:^|_)(?:TOKEN|KEY|APIKEY|KEY_ID|PASS|PASSWD|CREDENTIALS?)$|SECRET|PASSWORD"
)


def is_secret_name(name: str) -> bool:
    """Return True when the environment variable *name* can hold a secret."""
    return bool(_SECRET_NAME.search(name.upper()))


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
        self._log_file: TextIO | None = None

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
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=self.shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
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

    def metadata(self, base_dir: Path | None = None) -> dict[str, Any]:
        """Return the server record. ``log_path`` is relative to *base_dir*.

        Without *base_dir*, or when the log is outside it, ``log_path`` is
        the file name only. The record thus holds no home directory path.
        """
        environment = {
            key: "<redacted>" if is_secret_name(key) else value
            for key, value in self.environment.items()
        }
        try:
            log_path = str(self.log_path.relative_to(base_dir)) if base_dir else self.log_path.name
        except ValueError:
            log_path = self.log_path.name
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
            "log_path": log_path,
        }

    def save_metadata(self, path: Path) -> None:
        metadata = self.metadata(base_dir=Path(path).parent)
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def __enter__(self) -> VllmServer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
