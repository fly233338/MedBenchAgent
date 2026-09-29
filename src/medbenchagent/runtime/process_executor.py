"""Managed subprocess execution for preparation workers and adapters."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessReceipt:
    argv: tuple[str, ...]
    returncode: int | None
    started_at: float
    finished_at: float
    timed_out: bool = False
    cancelled: bool = False


class ProcessExecutor:
    def __init__(self, *, timeout: float | None = None) -> None:
        self.timeout = timeout
        self.process: subprocess.Popen[str] | None = None

    def run(self, argv, *, cwd=None, env=None, stdout=None, stderr=None, stop_requested=None):
        started = time.time()
        self.process = subprocess.Popen(
            [str(x) for x in argv], cwd=cwd, env=env, stdout=stdout, stderr=stderr,
            text=True, encoding="utf-8", start_new_session=os.name != "nt",
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
        )
        timed_out = cancelled = False
        deadline = started + self.timeout if self.timeout is not None else None
        while self.process.poll() is None:
            if stop_requested is not None and stop_requested():
                cancelled = True
                self.terminate()
                break
            if deadline is not None and time.time() >= deadline:
                timed_out = True
                self.terminate()
                break
            time.sleep(0.1)
        returncode = self.process.wait()
        return ProcessReceipt(tuple(str(x) for x in argv), returncode, started, time.time(), timed_out, cancelled)

    def terminate(self, *, kill_after: float = 5.0) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=kill_after)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
