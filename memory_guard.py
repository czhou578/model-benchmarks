#!/usr/bin/env python3
"""Run a benchmark command and stop it when system memory runs low.

On a unified-memory machine, the GPU and the CPU share one memory pool.
For example, the DGX Spark has 121 GB of unified memory, and nvidia-smi
reports its GPU memory as [N/A]. This guard reads /proc/meminfo:

- It refuses to start when MemAvailable is below --min-start-gib.
- It samples memory every --interval-s seconds into a CSV file.
- It sends SIGTERM to the command when MemAvailable stays below
  --floor-gib for --grace-samples samples in a row. core_runner.py
  handles SIGTERM and stops its managed vLLM server.
- It puts the CSV and a memory_guard.json summary into the run
  directory that core_runner.py prints. Without --csv, it writes the
  samples to a temporary file and moves that file into the run directory.
  With --csv, it keeps the file at that path and copies it.

Usage:
    python memory_guard.py --floor-gib 8 -- \\
        python core_runner.py --model models/qwen3.8_27b_nvidia_nvfp4.yml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

GIB = 1024**3
# core_runner.py prints an absolute path, and the path can hold spaces.
RUN_DIR_PATTERN = re.compile(r"writing results to (.+?)\s*$")
CSV_FIELDS = [
    "elapsed_s",
    "timestamp",
    "mem_total_gib",
    "mem_available_gib",
    "mem_used_gib",
    "swap_used_gib",
]
EXIT_LOW_MEMORY = 3


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[memory_guard {ts}] {msg}", flush=True)


def parse_meminfo(text: str) -> dict[str, float]:
    """Return the /proc/meminfo fields that the guard needs, in GiB."""
    kib: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            kib[key] = int(parts[0])
    total = kib["MemTotal"] * 1024 / GIB
    available = kib["MemAvailable"] * 1024 / GIB
    swap_used = (kib.get("SwapTotal", 0) - kib.get("SwapFree", 0)) * 1024 / GIB
    return {
        "mem_total_gib": round(total, 2),
        "mem_available_gib": round(available, 2),
        "mem_used_gib": round(total - available, 2),
        "swap_used_gib": round(swap_used, 2),
    }


def read_memory() -> dict[str, float]:
    return parse_meminfo(Path("/proc/meminfo").read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--floor-gib", type=float, default=8.0,
                        help="Stop the command below this MemAvailable (default 8)")
    parser.add_argument("--warn-gib", type=float, default=16.0,
                        help="Log a warning below this MemAvailable (default 16)")
    parser.add_argument("--min-start-gib", type=float, default=40.0,
                        help="Refuse to start below this MemAvailable (default 40)")
    parser.add_argument("--interval-s", type=float, default=5.0,
                        help="Seconds between two memory samples (default 5)")
    parser.add_argument("--grace-samples", type=int, default=3,
                        help="Low samples in a row before the guard stops the command (default 3)")
    parser.add_argument("--report-every-s", type=float, default=300.0,
                        help="Log the memory state at this interval (default 300)")
    parser.add_argument("--csv", default=None,
                        help="Also keep all samples in this CSV file. Without it, the "
                             "guard writes a temporary file and moves it into the run "
                             "directory as memory_samples.csv")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="The command to run, after --")
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command to run after --")

    start_mem = read_memory()
    if start_mem["mem_available_gib"] < args.min_start_gib:
        log(f"refusing to start: {start_mem['mem_available_gib']} GiB available, "
            f"{args.min_start_gib} GiB required. Stop other GPU work first.")
        return EXIT_LOW_MEMORY
    log(f"start: {start_mem['mem_available_gib']} of {start_mem['mem_total_gib']} GiB "
        f"available; floor {args.floor_gib} GiB, warning {args.warn_gib} GiB")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        fd, temp_name = tempfile.mkstemp(prefix="memory_samples_", suffix=".csv")
        os.close(fd)
        csv_path = Path(temp_name)
    run_dir: list[Path] = []

    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    def pump_output() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            match = RUN_DIR_PATTERN.search(line)
            if match and not run_dir:
                run_dir.append(Path(match.group(1)))

    reader = threading.Thread(target=pump_output, daemon=True)
    reader.start()

    def forward(signum, _frame) -> None:
        log(f"received signal {signum}; forwarding SIGTERM to the command")
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    started = time.time()
    last_report = started
    low_count = 0
    min_available = start_mem["mem_available_gib"]
    stopped_for_memory = False

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        while proc.poll() is None:
            mem = read_memory()
            now = time.time()
            writer.writerow({
                "elapsed_s": round(now - started, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **mem,
            })
            fh.flush()

            available = mem["mem_available_gib"]
            min_available = min(min_available, available)
            if available < args.floor_gib:
                low_count += 1
                log(f"LOW MEMORY: {available} GiB available "
                    f"({low_count}/{args.grace_samples})")
                if low_count >= args.grace_samples and not stopped_for_memory:
                    log("stopping the command to protect the system")
                    stopped_for_memory = True
                    proc.send_signal(signal.SIGTERM)
            else:
                low_count = 0
                if available < args.warn_gib:
                    log(f"warning: {available} GiB available")

            if now - last_report >= args.report_every_s:
                log(f"{available} GiB available, {mem['mem_used_gib']} GiB used, "
                    f"minimum so far {min_available} GiB")
                last_report = now

            deadline = now + args.interval_s
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.2)

        if stopped_for_memory:
            try:
                proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                proc.kill()

    reader.join(timeout=5)
    log(f"command exited with code {proc.returncode}; "
        f"minimum available memory {min_available} GiB")

    if run_dir and run_dir[0].is_dir():
        target = run_dir[0] / "memory_samples.csv"
        if args.csv:
            shutil.copy2(csv_path, target)
        else:
            shutil.move(csv_path, target)
        (run_dir[0] / "memory_guard.json").write_text(json.dumps({
            "floor_gib": args.floor_gib,
            "warn_gib": args.warn_gib,
            "interval_s": args.interval_s,
            "start_available_gib": start_mem["mem_available_gib"],
            "min_available_gib": min_available,
            "stopped_for_memory": stopped_for_memory,
            "return_code": proc.returncode,
        }, indent=2) + "\n")
        log(f"memory samples written to {target}")
    elif not args.csv:
        # Keep the samples: without a run dir, the temporary file is the only copy.
        log(f"no run directory found; memory samples stay in {csv_path}")

    if stopped_for_memory:
        return EXIT_LOW_MEMORY
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
