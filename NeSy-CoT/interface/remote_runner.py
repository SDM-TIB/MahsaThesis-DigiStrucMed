"""Run a pipeline step on a VM, inside tmux, so it survives SSH disconnects
and the local laptop shutting down.

Mirrors the exact, already-proven shape of README_VM.md sections 1 and 5-8
(status.txt + ERR trap, tee'd per-step log, tar+sha256 archive, checksum
verification on download) but generated per-step and driven over paramiko
instead of pasted manually.
"""
from __future__ import annotations

import re
import shlex
import time
from pathlib import Path
from typing import Optional

import config_utils
import pipeline_steps
import run_store
from pipeline_steps import PROJECT_DIR, STEPS
from ssh_client import SSHClient, SSHError

_ETA_RE = re.compile(r"(\d+)/(\d+)\s*\[(\d+):(\d\d)<(\d+):(\d\d)")


def resolve_root(ssh: SSHClient) -> str:
    home = ssh.run_checked("echo $HOME").strip()
    return f"{home}/{pipeline_steps.REMOTE_ROOT_NAME}"


def _local_and_remote(path: Path, root: str) -> tuple[Path, str]:
    if path.is_absolute():
        try:
            rel = path.relative_to(PROJECT_DIR).as_posix()
        except ValueError:
            rel = path.name
        return path, f"{root}/{rel}"
    return PROJECT_DIR / path, f"{root}/{path.as_posix()}"


def test_connection(ssh: SSHClient) -> str:
    status, out, err = ssh.run("nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader")
    if status != 0:
        return f"Connected, but nvidia-smi failed:\n{err or out}"
    return f"Connected. GPU: {out.strip()}"


def stage_files_for_step(
    ssh: SSHClient,
    root: str,
    step_id: str,
    cfg: dict,
    cot_version: Optional[str],
    config_local_path: Path,
) -> list[str]:
    step = STEPS[step_id]
    uploaded: list[str] = []

    if step_id == "install":
        ssh.upload_file(PROJECT_DIR / "requirements.txt", f"{root}/requirements.txt")
        return ["requirements.txt"]

    if step.script:
        ssh.upload_file(PROJECT_DIR / step.script, f"{root}/{step.script}")
        uploaded.append(step.script)
    for code_file in step.code_files:
        ssh.upload_file(PROJECT_DIR / code_file, f"{root}/{code_file}")
        uploaded.append(code_file)

    ssh.upload_file(config_local_path, f"{root}/config.json")
    uploaded.append("config.json")

    if step.data_paths:
        for p in step.data_paths(cfg, cot_version):
            local, remote_path = _local_and_remote(p, root)
            if not local.exists():
                raise FileNotFoundError(
                    f"Required input not found locally: {local} (needed by step '{step_id}'). "
                    "Run the earlier pipeline stage first."
                )
            if local.is_dir():
                n = ssh.upload_dir(local, remote_path)
                uploaded.append(f"{remote_path.split(root + '/', 1)[-1]}/ ({n} files)")
            else:
                ssh.upload_file(local, remote_path)
                uploaded.append(remote_path.split(root + "/", 1)[-1])
    return uploaded


def _preamble(root: str, run_id: str, step_id: str, needs_token: bool) -> str:
    token_setup = f'export HF_TOKEN="$(cat "{root}/.hf_token")"\n' if needs_token else ""
    token_cleanup = f'rm -f "{root}/.hf_token"; ' if needs_token else ""
    label = STEPS[step_id].label
    return f"""set -Eeuo pipefail
RUN_DIR="{root}/logs/{run_id}"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/{step_id}.log") 2>&1
trap 'rc=$?; echo "FAILED at line $LINENO, exit $rc"; echo "FAILED exit=$rc" > "$RUN_DIR/status.txt"; {token_cleanup}exit "$rc"' ERR
trap '{token_cleanup}true' EXIT
echo RUNNING > "$RUN_DIR/status.txt"
echo '=== {label} ==='
date -u
cd {root}
{token_setup}export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
"""


def build_run_script(root: str, run_id: str, step_id: str, cot_version: Optional[str]) -> str:
    step = STEPS[step_id]
    needs_token = step.needs_gpu  # step1/2/3 call utils.login_huggingface()
    pre = _preamble(root, run_id, step_id, needs_token)
    if step_id == "install":
        body = pipeline_steps.REMOTE_INSTALL_SCRIPT.replace("__ROOT__", root)
    else:
        args = " ".join(shlex.quote(a) for a in (step.build_args("config.json", cot_version) if step.build_args else []))
        body = f"source .venv/bin/activate\npython -u {step.script} {args}\n"
    footer = '\necho SUCCESS > "$RUN_DIR/status.txt"\ndate -u\n'
    return pre + body + footer


def _clear_remote_adapter_dirs(ssh: SSHClient, root: str, cfg: dict, step_id: str,
                               cot_version: Optional[str]) -> list[str]:
    """rm -rf a training step's adapter dirs on the VM so it trains fresh
    instead of resuming a leftover checkpoint. Output dirs are config-relative
    (e.g. ./outputs/qKG/finetuned_...), rooted under REMOTE_ROOT."""
    cleared = []
    out_rel = str(Path(cfg["output_dir"]).as_posix()).lstrip("./")
    tag = cfg["model_key"].replace("-", "_")
    names = []
    if step_id == "step2":
        names = [f"finetuned_{tag}_baseline"]
    elif step_id == "step3":
        versions = ["CoT2", "CoT3"] if cot_version in (None, "both") else [cot_version]
        names = [f"finetuned_{tag}_with_rules_{v}" for v in versions]
    for name in names:
        remote_dir = f"{root}/{out_rel}/{name}"
        ssh.run(f"rm -rf {shlex.quote(remote_dir)}")
        cleared.append(remote_dir)
    return cleared


def _preflight_hf(ssh: SSHClient, root: str, model_name: str) -> None:
    """Validate the just-uploaded token BEFORE launching a GPU run, so an
    invalid token (401) or missing gated-model access (403) is reported
    immediately in the app instead of as a traceback in the VM log minutes
    later. Uses curl (no venv needed). Best-effort: only hard-fails on a
    definitive 401/403, never on a flaky/unreachable check."""
    token_file = shlex.quote(f"{root}/.hf_token")

    def http_code(url: str) -> str:
        cmd = (
            f'curl -s -o /dev/null -m 20 -w "%{{http_code}}" '
            f'-H "Authorization: Bearer $(cat {token_file})" {shlex.quote(url)}'
        )
        _status, out, _err = ssh.run(cmd, timeout=30)
        return out.strip()

    who = http_code("https://huggingface.co/api/whoami-v2")
    if who == "401":
        raise ValueError(
            "Hugging Face token is invalid (401 from whoami). Re-copy a valid "
            "Read token from https://huggingface.co/settings/tokens into the "
            "sidebar 'Hugging Face token' field, then run again."
        )
    if who != "200":
        return  # curl/network unavailable — don't block the run on a flaky check

    access = http_code(f"https://huggingface.co/{model_name}/resolve/main/config.json")
    if access == "403":
        raise ValueError(
            f"Token is valid but has no access to the gated model '{model_name}' "
            f"(403). Request access at https://huggingface.co/{model_name} and "
            "accept the license (approval is usually quick), then run again."
        )
    if access == "401":
        raise ValueError(
            "Hugging Face token was rejected (401) when reaching the model. "
            "Re-enter a valid Read token in the sidebar."
        )


def start_remote_step(
    ssh: SSHClient,
    host: str,
    username: str,
    kg: str,
    step_id: str,
    cfg: dict,
    cot_version: Optional[str] = None,
    hf_token: Optional[str] = None,
    fresh: bool = False,
) -> str:
    root = resolve_root(ssh)
    run_id = run_store.new_run_id(step_id)

    local_run_dir = run_store.RUNS_DIR / "remote" / run_id
    config_local_path = config_utils.materialize_config(cfg, local_run_dir / "config.used.json")

    cleared = _clear_remote_adapter_dirs(ssh, root, cfg, step_id, cot_version) if fresh else []

    uploaded = stage_files_for_step(ssh, root, step_id, cfg, cot_version, config_local_path)

    step = STEPS[step_id]
    if step.needs_gpu:
        if not hf_token:
            raise ValueError(f"Step '{step_id}' needs a Hugging Face token.")
        ssh.upload_text(hf_token.strip(), f"{root}/.hf_token")
        ssh.run_checked(f'chmod 600 "{root}/.hf_token"')
        # Fail fast on a bad token / missing model access before launching.
        try:
            _preflight_hf(ssh, root, cfg["available_models"][cfg["model_key"]])
        except ValueError:
            ssh.run(f'rm -f {shlex.quote(f"{root}/.hf_token")}')
            raise

    script_text = build_run_script(root, run_id, step_id, cot_version)
    remote_script_path = f"{root}/logs/{run_id}/run.sh"
    ssh.upload_text(script_text, remote_script_path)

    tmux_session = f"nesycot-{run_id}"
    ssh.start_background(tmux_session, remote_script_path)

    run_store.upsert_remote_run(
        run_id,
        {
            "step_id": step_id,
            "label": step.label,
            "kg": kg,
            "cot_version": cot_version,
            "host": host,
            "username": username,
            "remote_root": root,
            "remote_run_dir": f"{root}/logs/{run_id}",
            "tmux_session": tmux_session,
            "status": "RUNNING",
            "started_at": time.time(),
            "uploaded_files": uploaded,
            "config_path": str(config_local_path),
            "fresh": fresh,
            "cleared_dirs": cleared,
        },
    )
    return run_id


def refresh_status(ssh: SSHClient, run_id: str) -> Optional[dict]:
    record = run_store.get_remote_run(run_id)
    if record is None:
        return None
    status_text = ssh.read_remote_file(f"{record['remote_run_dir']}/status.txt")
    if status_text:
        status_text = status_text.strip()
        if status_text.startswith("FAILED"):
            record["status"] = "FAILED"
        elif status_text == "SUCCESS":
            record["status"] = "SUCCESS"
        elif status_text == "STOPPED":
            record["status"] = "STOPPED"
        else:
            record["status"] = "RUNNING"
    if record["status"] == "RUNNING" and not ssh.tmux_session_exists(record["tmux_session"]):
        # tmux session gone but status.txt never flipped — the process died
        # (VM restarted, OOM-killed outside the trap, etc.)
        record["status"] = "FAILED"
        record["status_detail"] = "tmux session ended without a status.txt update"
    run_store.upsert_remote_run(run_id, record)
    return record


def stop_run(ssh: SSHClient, run_id: str) -> str:
    """Kill the tmux session running a step (equivalent to the VM losing
    power mid-job) and mark status.txt STOPPED so refresh_status doesn't
    report it as a crash."""
    record = run_store.get_remote_run(run_id)
    if record is None:
        return "Run not found."
    if record.get("status") != "RUNNING":
        return f"Run is already {record['status']}."

    session = record["tmux_session"]
    if ssh.tmux_session_exists(session):
        ssh.run(f"tmux kill-session -t {shlex.quote(session)}")
    ssh.upload_text("STOPPED\n", f"{record['remote_run_dir']}/status.txt")

    record["status"] = "STOPPED"
    record["ended_at"] = time.time()
    run_store.upsert_remote_run(run_id, record)
    return "Stopped (tmux session killed; any checkpoint already saved is intact)."


def tail_log(ssh: SSHClient, run_id: str, lines: int = 200) -> str:
    record = run_store.get_remote_run(run_id)
    if record is None:
        return ""
    log_path = f"{record['remote_run_dir']}/{record['step_id']}.log"
    return ssh.tail_remote_file(log_path, lines)


def parse_eta(log_tail: str) -> Optional[str]:
    """Pull HF Trainer's own tqdm ETA off the last matching progress line."""
    matches = _ETA_RE.findall(log_tail)
    if not matches:
        return None
    step, total, _eh, _em, rh, rm = matches[-1]
    return f"step {step}/{total} — ETA {rh}:{rm}"


def gpu_snapshot(ssh: SSHClient) -> str:
    _status, out, err = ssh.run("nvidia-smi")
    return out or err


def list_vm_files(ssh: SSHClient, run_id: str) -> str:
    """List what's currently in ~/NeSyCoT-final on the VM — specifically the
    outputs/<kg> and logs that 'Archive & download' will bundle — so you can
    see what exists before archiving, mirroring the manual `ls -lh` step."""
    record = run_store.get_remote_run(run_id)
    if record is None:
        return "(run not found)"
    root = record["remote_root"]
    kg = record["kg"]
    out_rel = f"outputs/{kg}" if kg in ("guidelineKG", "qKG") else "outputs"
    run_dir = record["remote_run_dir"]
    listing = f"""
echo '=== results in {out_rel}/ (bundled on download) ==='
ls -lah {shlex.quote(f'{root}/{out_rel}')} 2>/dev/null || echo '(no outputs yet)'
echo
echo '=== result JSONs ==='
ls -1 {shlex.quote(f'{root}/{out_rel}')}/*.json 2>/dev/null || echo '(none yet)'
echo
echo '=== LoRA adapter dirs (size each) ==='
du -sh {shlex.quote(f'{root}/{out_rel}')}/finetuned_* 2>/dev/null || echo '(none yet)'
echo
echo '=== logs for this run: {run_dir} ==='
ls -lah {shlex.quote(run_dir)} 2>/dev/null || echo '(no logs yet)'
echo
echo '=== total size that would be archived ==='
du -sh {shlex.quote(f'{root}/{out_rel}')} {shlex.quote(f'{root}/logs')} 2>/dev/null || echo '(nothing yet)'
"""
    _status, out, err = ssh.run(f"bash -c {shlex.quote(listing)}")
    return out or err or "(no output)"


def archive_and_download(ssh: SSHClient, run_id: str, local_dest_dir: Path) -> Path:
    record = run_store.get_remote_run(run_id)
    if record is None:
        raise SSHError(f"Unknown run_id: {run_id}")
    root = record["remote_root"]
    kg = record["kg"]
    output_rel = f"outputs/{kg}" if kg in ("guidelineKG", "qKG") else "outputs"
    archive_cmd = f"""set -Eeuo pipefail
cd {root}
mkdir -p final_outputs
ARCHIVE="final_outputs/nesycot_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
tar -czf "$ARCHIVE" {output_rel} logs && sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
echo "$ARCHIVE" > final_outputs/latest_archive.txt
echo "$ARCHIVE"
"""
    out = ssh.run_checked(f"bash -c {shlex.quote(archive_cmd)}", timeout=600)
    archive_rel = out.strip().splitlines()[-1]
    archive_name = Path(archive_rel).name

    local_dest_dir.mkdir(parents=True, exist_ok=True)
    local_archive = local_dest_dir / archive_name
    local_sha_path = Path(str(local_archive) + ".sha256")
    ssh.download_file(f"{root}/{archive_rel}", local_archive)
    ssh.download_file(f"{root}/{archive_rel}.sha256", local_sha_path)

    expected = local_sha_path.read_text(encoding="utf-8").split()[0]
    import hashlib

    actual = hashlib.sha256(local_archive.read_bytes()).hexdigest()
    if actual != expected:
        raise SSHError(f"Checksum mismatch after download: expected {expected}, got {actual}")

    import tarfile

    with tarfile.open(local_archive) as tf:
        tf.extractall(local_dest_dir)

    return local_archive
