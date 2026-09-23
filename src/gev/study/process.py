"""Isolated study child-process execution, output logging, and progress."""

from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path


_STEP = re.compile(r"\bstep\s+(\d+)/(\d+)\s+loss\s+([-+0-9.eE]+)")


def run_child(command, *, log_path: Path, on_update, heartbeat_interval=15.0,
              progress_label="child", stage=None, clock=time.monotonic):
    """Run a child while teeing its combined output to disk and reporting progress."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = clock()
    last_report = started
    latest = {"step": None, "total_steps": None, "latest_loss": None}
    tail = bytearray()
    chunks: queue.Queue = queue.Queue(maxsize=32)
    sentinel = object()
    child_command = [command[0], "-u", *command[1:]] if command[1:3] == ["-m", "gev"] else command
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = None
    reader = None
    read_error = []

    def read_output(stream):
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                chunks.put(chunk)
        except BaseException as exc:  # propagate reader failures to the owner thread
            read_error.append(exc)
        finally:
            chunks.put(sentinel)

    try:
        with log_path.open("xb") as logfile:
            process = subprocess.Popen(child_command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, bufsize=0, env=environment)
            reader = threading.Thread(target=read_output, args=(process.stdout,), daemon=True)
            reader.start()
            line_buffer = bytearray()
            reader_done = False
            while not reader_done:
                try:
                    chunk = chunks.get(timeout=min(0.25, heartbeat_interval))
                except queue.Empty:
                    chunk = None
                now = clock()
                if chunk is sentinel:
                    reader_done = True
                elif chunk:
                    logfile.write(chunk)
                    logfile.flush()
                    tail.extend(chunk)
                    if len(tail) > 8192:
                        del tail[:-8192]
                    progress_changed = False
                    for byte in chunk:
                        if byte == 10:
                            line = bytes(line_buffer).decode("utf-8", errors="replace").strip()
                            line_buffer.clear()
                            match = _STEP.search(line)
                            if match:
                                latest.update(step=int(match.group(1)), total_steps=int(match.group(2)),
                                              latest_loss=float(match.group(3)))
                                progress_changed = True
                        elif len(line_buffer) < 16384:
                            line_buffer.append(byte)
                    if progress_changed:
                        on_update(latest, now - started, False)
                        if now - last_report >= 1.0:
                            print(f"[{progress_label}] step {latest['step']}/{latest['total_steps']} "
                                  f"loss {latest['latest_loss']:.6g} elapsed {int(now - started)}s", flush=True)
                            last_report = now
                if now - last_report >= heartbeat_interval:
                    on_update(latest, now - started, True)
                    if stage == "train":
                        state = (f"step {latest['step']}/{latest['total_steps']}; "
                                 if latest["step"] is not None else "loading; ")
                        message = f"{state}still running"
                    else:
                        message = "working"
                    print(f"[{progress_label}] {message}; elapsed {int(now - started)}s", flush=True)
                    last_report = now
            if line_buffer:
                match = _STEP.search(line_buffer.decode("utf-8", errors="replace"))
                if match:
                    latest.update(step=int(match.group(1)), total_steps=int(match.group(2)),
                                  latest_loss=float(match.group(3)))
            reader.join()
            logfile.flush()
            returncode = process.wait()
            if read_error:
                raise read_error[0]
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if reader is not None:
            while reader.is_alive():
                try:
                    chunks.get(timeout=0.1)
                except queue.Empty:
                    pass
            reader.join()
        raise
    except Exception:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if reader is not None:
            while reader.is_alive():
                try:
                    chunks.get(timeout=0.1)
                except queue.Empty:
                    pass
            reader.join()
        raise

    excerpt = tail.decode("utf-8", errors="replace").strip()
    return {"returncode": returncode, **latest, "elapsed_seconds": clock() - started,
            "log_path": str(log_path), "output_excerpt": excerpt}
