"""Run a pipeline step locally as a fully detached background process.

Each step is wrapped in a small batch/shell script that writes its exit
code to a marker file when done — that marker (not a held-in-memory
subprocess handle) is the source of truth, so status survives the
Streamlit *server* being restarted, not just a browser tab being closed.
The HF token is passed through the process environment only, never written
to the wrapper script or any file on disk.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import config_utils
import pipeline_steps
import run_store
from pipeline_steps import PROJECT_DIR, STEPS


def _clear_adapter_dirs(cfg: dict, step_id: str, cot_version: Optional[str]) -> list[str]:
    """Delete a training step's leftover adapter/checkpoint dirs so it trains
    fresh instead of auto-resuming. Guarded: only removes paths that actually
    sit under the run's output_dir, never an arbitrary path."""
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir = output_dir.resolve()
    cleared = []
    for d in pipeline_steps.adapter_output_dirs(cfg, step_id, cot_version):
        d = d if d.is_absolute() else (PROJECT_DIR / d)
        d = d.resolve()
        if output_dir in d.parents and d.exists():
            shutil.rmtree(d)
            cleared.append(str(d))
    return cleared


def _deque_tail(lines: list[str], n: int) -> list[str]:
    return lines[-n:]


def start_local_step(
    step_id: str,
    kg: str,
    cfg: dict,
    cot_version: Optional[str] = None,
    hf_token: Optional[str] = None,
    fresh: bool = False,
) -> str:
    step = STEPS[step_id]
    run_id = run_store.new_run_id(step_id)
    run_dir = run_store.local_run_dir(run_id)

    cleared = _clear_adapter_dirs(cfg, step_id, cot_version) if fresh else []

    config_path = config_utils.materialize_config(cfg, run_dir / "config.used.json")
    log_path = run_dir / "step.log"
    exitcode_path = run_dir / "exitcode.txt"

    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
    env["WANDB_DISABLED"] = "true"
    env["PYTHONUNBUFFERED"] = "1"

    if step_id == "install":
        argv = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    else:
        assert step.script is not None
        args = step.build_args(str(config_path), cot_version) if step.build_args else []
        argv = [sys.executable, str(PROJECT_DIR / step.script), *args]

    if os.name == "nt":
        bat_path = run_dir / "run.bat"
        quoted = " ".join(f'"{a}"' if " " in a else a for a in argv)
        bat_path.write_text(
            "@echo off\r\n"
            f'cd /d "{PROJECT_DIR}"\r\n'
            f'{quoted} > "{log_path}" 2>&1\r\n'
            f'echo %errorlevel% > "{exitcode_path}"\r\n',
            encoding="utf-8",
        )
        # CREATE_NO_WINDOW hides the console but keeps std handles intact, so
        # the .bat file's own `>` redirection still works; DETACHED_PROCESS
        # (tried first) strips those handles and silently produces an empty
        # log despite the step succeeding. A plain Popen child already
        # outlives the parent Streamlit process on Windows without it.
        proc = subprocess.Popen(
            ["cmd", "/c", str(bat_path)],
            cwd=str(PROJECT_DIR),
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW,
            close_fds=True,
        )
    else:
        sh_path = run_dir / "run.sh"
        quoted = " ".join(f"'{a}'" for a in argv)
        sh_path.write_text(
            "#!/bin/sh\n"
            f"cd '{PROJECT_DIR}'\n"
            f"{quoted} > '{log_path}' 2>&1\n"
            f"echo $? > '{exitcode_path}'\n",
            encoding="utf-8",
        )
        sh_path.chmod(0o755)
        proc = subprocess.Popen(
            ["/bin/sh", str(sh_path)],
            cwd=str(PROJECT_DIR),
            env=env,
            start_new_session=True,
            close_fds=True,
        )

    run_store.write_local_status(
        run_id,
        {
            "step_id": step_id,
            "label": step.label,
            "kg": kg,
            "cot_version": cot_version,
            "status": "RUNNING",
            "pid": proc.pid,
            "started_at": time.time(),
            "log_path": str(log_path),
            "config_path": str(config_path),
            "fresh": fresh,
            "cleared_dirs": cleared,
        },
    )
    return run_id


def refresh_status(run_id: str) -> Optional[dict]:
    status = run_store.read_local_status(run_id)
    if status is None:
        return None
    if status.get("status") != "RUNNING":
        return status

    run_dir = run_store.LOCAL_RUNS_DIR / run_id
    exitcode_path = run_dir / "exitcode.txt"
    if exitcode_path.exists():
        try:
            code = int(exitcode_path.read_text(encoding="utf-8").strip())
        except ValueError:
            code = 1
        status["status"] = "SUCCESS" if code == 0 else "FAILED"
        status["exit_code"] = code
        status["ended_at"] = time.time()
        run_store.write_local_status(run_id, status)
    return status


def stop_run(run_id: str) -> str:
    """Kill the whole process tree for a running local step.

    ``status["pid"]`` is the wrapper (cmd.exe/sh) pid, not the real
    python.exe pid, so a plain terminate() would leave the actual script
    (and, for "install", any pip subprocess) running — hence /T on Windows
    and killpg on POSIX, both of which target the whole tree/group.
    """
    status = run_store.read_local_status(run_id)
    if status is None:
        return "Run not found."
    if status.get("status") != "RUNNING":
        return f"Run is already {status['status']}."

    pid = status.get("pid")
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True, text=True,
        )
        detail = result.stdout.strip() or result.stderr.strip()
    else:
        import signal
        try:
            os.killpg(pid, signal.SIGTERM)
            detail = f"SIGTERM sent to process group {pid}"
        except ProcessLookupError:
            detail = "Process had already exited"

    status["status"] = "STOPPED"
    status["ended_at"] = time.time()
    status["status_detail"] = detail
    run_store.write_local_status(run_id, status)
    return detail


def tail_log(run_id: str, n: int = 200) -> str:
    status = run_store.read_local_status(run_id)
    if status is None:
        return ""
    log_path = Path(status["log_path"])
    if not log_path.exists():
        return ""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(_deque_tail(lines, n))
