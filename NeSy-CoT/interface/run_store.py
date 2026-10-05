"""Run bookkeeping so the UI can reload state after a page refresh or restart.

Local runs are fully described by files under runs/local/<run_id>/ (status,
logs). Remote runs live on the VM (status.txt / logs under
~/NeSyCoT-final/logs/<run_id>/, inside a tmux session that keeps running
regardless of the local machine) — runs/remote_index.json is only a small,
credential-free pointer ({run_id, host, remote dir, tmux session name}) so
the Streamlit app can find and re-attach to it after restarting.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent / "runs"
LOCAL_RUNS_DIR = RUNS_DIR / "local"
REMOTE_INDEX_PATH = RUNS_DIR / "remote_index.json"


def new_run_id(step_id: str) -> str:
    return f"{step_id}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


# -- local runs --------------------------------------------------------------

def local_run_dir(run_id: str) -> Path:
    d = LOCAL_RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_local_status(run_id: str, status: dict) -> None:
    status = {**status, "run_id": run_id, "updated_at": time.time()}
    (local_run_dir(run_id) / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def read_local_status(run_id: str) -> dict | None:
    path = LOCAL_RUNS_DIR / run_id / "status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def list_local_runs() -> list[dict]:
    if not LOCAL_RUNS_DIR.exists():
        return []
    runs = []
    for d in LOCAL_RUNS_DIR.iterdir():
        status = read_local_status(d.name)
        if status:
            runs.append(status)
    return sorted(runs, key=lambda r: r.get("started_at", 0), reverse=True)


# -- remote runs --------------------------------------------------------------

def _load_remote_index() -> dict:
    if not REMOTE_INDEX_PATH.exists():
        return {}
    try:
        return json.loads(REMOTE_INDEX_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_remote_index(index: dict) -> None:
    REMOTE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    REMOTE_INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")


def upsert_remote_run(run_id: str, record: dict) -> None:
    index = _load_remote_index()
    index[run_id] = {**index.get(run_id, {}), **record, "run_id": run_id, "updated_at": time.time()}
    _save_remote_index(index)


def get_remote_run(run_id: str) -> dict | None:
    return _load_remote_index().get(run_id)


def list_remote_runs() -> list[dict]:
    return sorted(_load_remote_index().values(), key=lambda r: r.get("started_at", 0), reverse=True)
