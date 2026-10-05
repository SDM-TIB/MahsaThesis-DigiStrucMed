"""NeSy-CoT pipeline control panel.

Run with:
    streamlit run NeSy-CoT/interface/app.py

One tab per pipeline stage (install libs -> upload rules -> CoT generation
-> prepare data -> fine-tune steps 1-3), runnable locally or on a VM, for
either guidelineKG or qKG. See ../README_VM.md for the manual process this
automates and the plan this was built from for design rationale.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config_utils
import local_runner
import pipeline_steps
import remote_runner
import run_store
import vm_profiles
from pipeline_steps import PROJECT_DIR, STEPS
from ssh_client import SSHClient, SSHError

st.set_page_config(page_title="NeSy-CoT Pipeline", layout="wide")

STATUS_BADGE = {"RUNNING": "🟡", "SUCCESS": "🟢", "FAILED": "🔴", "STOPPED": "⏹️"}
REQUIRED_RULE_COLUMNS = {"Body", "Head"}
CONFIDENCE_COLUMNS = ["Pca_Confidence", "PCA_Confidence", "PCA Confidence", "pca_confidence", "PCA"]


def validate_rules_csv(df: pd.DataFrame) -> None:
    missing = REQUIRED_RULE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")
    if not any(c in df.columns for c in CONFIDENCE_COLUMNS):
        raise ValueError(
            "No confidence column found — expected one of: " + ", ".join(CONFIDENCE_COLUMNS)
        )


# --------------------------------------------------------------------------
# Sidebar — global setup
# --------------------------------------------------------------------------

st.sidebar.header("Setup")

kg = st.sidebar.selectbox("Knowledge graph", ["guidelineKG", "qKG"])
base_cfg = config_utils.load_base_config(kg)

model_keys = list(base_cfg["available_models"].keys())
model_key = st.sidebar.selectbox(
    "Model", model_keys, index=model_keys.index(base_cfg["model_key"]) if base_cfg["model_key"] in model_keys else 0
)

smoke = st.sidebar.toggle(
    "Smoke test", value=False,
    help="Runs a tiny, fast version of whichever step you click (10 training steps, "
         "a handful of eval samples) so you can catch config/environment errors "
         "before committing GPU time to a full run.",
)

hf_token = st.sidebar.text_input(
    "Hugging Face token", type="password",
    help="Required for fine-tuning steps 1-3. Kept in memory for this session only, "
         "never written to disk.",
).strip()  # strip stray whitespace/newlines from a paste before it's ever used

target = st.sidebar.radio("Run on", ["Local", "VM"], horizontal=True)

vm_profile = None
if target == "VM":
    if "vm_profiles" not in st.session_state:
        st.session_state.vm_profiles = vm_profiles.load_profiles()
    profiles = st.session_state.vm_profiles

    def _label(p: dict) -> str:
        badge = "" if p.get("verified") else " ⚠ gpu_max_memory_mb unverified"
        return (
            f"{p['label']} — {p['vram_gb']}GB VRAM, {p['ram_gb']}GB RAM, "
            f"{p['disk_gb']}GB disk, ${p['price_per_hr']}/hr{badge}"
        )

    default_idx = next((i for i, p in enumerate(profiles) if p.get("is_default")), 0)
    profile_idx = st.sidebar.selectbox(
        "VM profile", range(len(profiles)), format_func=lambda i: _label(profiles[i]), index=default_idx
    )
    vm_profile = profiles[profile_idx]

    with st.sidebar.expander("Edit VM profiles"):
        edited = st.text_area(
            "Profiles JSON", value=json.dumps(profiles, indent=2), height=240, key="vm_profiles_editor"
        )
        if st.button("Save profiles"):
            try:
                new_profiles = json.loads(edited)
                vm_profiles.save_profiles(new_profiles)
                st.session_state.vm_profiles = new_profiles
                st.success("Saved.")
                st.rerun()
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                st.error(f"Invalid profiles JSON: {exc}")

    st.sidebar.subheader("VM connection")
    st.sidebar.caption("Entered fresh each session — never saved to disk.")
    vm_host = st.sidebar.text_input("Host / IP", key="vm_host")
    vm_port = st.sidebar.number_input("Port", value=22, min_value=1, max_value=65535, key="vm_port")
    vm_user = st.sidebar.text_input("Username", value="riftuser", key="vm_user")
    vm_password = st.sidebar.text_input("Password", type="password", key="vm_password")

    if st.sidebar.button("Test SSH connection", type="primary"):
        try:
            ssh = SSHClient(vm_host, vm_user, vm_password, int(vm_port))
            msg = remote_runner.test_connection(ssh)
            old = st.session_state.get("ssh")
            if old is not None:
                old.close()
            st.session_state.ssh = ssh
            st.session_state.ssh_info = {"host": vm_host, "username": vm_user}
            st.sidebar.success(msg)
        except SSHError as exc:
            st.sidebar.error(str(exc))

    if st.session_state.get("ssh") is not None:
        info = st.session_state.ssh_info
        st.sidebar.caption(f"Connected: {info['username']}@{info['host']}")

# Build the merged config for the current selection, then let the user edit
# the final JSON directly before running anything.
base_run_cfg = config_utils.build_run_config(kg, vm_profile=vm_profile, smoke=smoke, model_key=model_key)
editor_key = f"cfg_editor_{kg}_{vm_profile['id'] if vm_profile else 'local'}_{smoke}_{model_key}"
with st.sidebar.expander("Advanced: edit merged config"):
    edited_json = st.text_area(
        "config.json (used for the next run you start)",
        value=json.dumps(base_run_cfg, indent=2),
        height=300,
        key=editor_key,
    )
try:
    run_cfg = json.loads(st.session_state.get(editor_key, json.dumps(base_run_cfg)))
except json.JSONDecodeError as exc:
    st.sidebar.error(f"Invalid config JSON — falling back to defaults: {exc}")
    run_cfg = base_run_cfg


def _start_step(step_id: str, cot_version: str | None = None, fresh: bool = False) -> None:
    try:
        if target == "Local":
            run_id = local_runner.start_local_step(step_id, kg, run_cfg, cot_version, hf_token or None, fresh)
            st.session_state["last_run"] = ("local", run_id)
        else:
            ssh = st.session_state.get("ssh")
            if ssh is None:
                st.error("Test the SSH connection first (sidebar).")
                return
            run_id = remote_runner.start_remote_step(
                ssh, st.session_state.ssh_info["host"], st.session_state.ssh_info["username"],
                kg, step_id, run_cfg, cot_version, hf_token or None, fresh,
            )
            st.session_state["last_run"] = ("remote", run_id)
        st.success(f"Started: {run_id}")
    except (SSHError, FileNotFoundError, ValueError) as exc:
        st.error(str(exc))


def _run_config_summary(record: dict) -> str:
    """Read back the config actually materialized for this run, so it's
    obvious at a glance whether a run used the smoke config or the full
    one — rather than having to grep config.used.json by hand."""
    config_path = record.get("config_path")
    if not config_path or not Path(config_path).exists():
        return ""
    try:
        used = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    tcfg = used.get("training", {})
    ecfg = used.get("evaluation", {})
    is_smoke = str(used.get("output_dir", "")).rstrip("/\\").endswith("_smoke")
    tag = "SMOKE" if is_smoke else "FULL"
    return (
        f"{tag} · num_steps={tcfg.get('num_steps')} · eval_samples={ecfg.get('max_samples')} "
        f"· output_dir={used.get('output_dir')}"
    )


def _stop_run(kind: str, run_id: str) -> None:
    if kind == "Local" or kind == "local":
        detail = local_runner.stop_run(run_id)
    else:
        ssh = st.session_state.get("ssh")
        if ssh is None:
            st.error("Reconnect via the sidebar first — stopping a VM run needs an active SSH session.")
            return
        detail = remote_runner.stop_run(ssh, run_id)
    st.toast(f"{run_id}: {detail}")


def render_run_button(step_id: str, cot_version: str | None = None) -> None:
    step = STEPS[step_id]
    if step.needs_gpu and not hf_token:
        st.warning("Enter a Hugging Face token in the sidebar — this step fine-tunes/evaluates a model.")

    all_runs = run_store.list_local_runs() if target == "Local" else run_store.list_remote_runs()
    runs = [r for r in all_runs if r["step_id"] == step_id][:5]
    already_running = [r for r in runs if r["status"] == "RUNNING"]

    label = f"Run: {step.label}" + (f" [{cot_version}]" if cot_version else "")
    if already_running:
        st.warning(
            f"{len(already_running)} run(s) of this step are already RUNNING on {target} "
            f"(see below) — starting another will compete for the same GPU. "
            "Stop the existing run first, or confirm below to start anyway."
        )
        allow_duplicate = st.checkbox(
            "Start another instance anyway", key=f"force_{step_id}_{cot_version}_{target}"
        )
    else:
        allow_duplicate = True

    # Training steps (step2/step3) auto-resume from any leftover checkpoint in
    # the output dir — desirable for real runs (resume after an interruption),
    # but for a smoke re-run it means "resume a finished checkpoint and skip
    # training". Default the clear ON in smoke so smoke tests actually retrain,
    # OFF otherwise so an interrupted real run can still resume.
    fresh = False
    if pipeline_steps.adapter_output_dirs(run_cfg, step_id, cot_version):
        fresh = st.checkbox(
            "Fresh start (clear old checkpoints before training)",
            value=smoke,
            key=f"fresh_{step_id}_{cot_version}_{target}",
            help="ON: delete this step's previous adapter/checkpoints so it trains from scratch. "
                 "OFF: resume from the latest checkpoint if one exists (saves a re-interrupted run). "
                 "Defaults ON in smoke mode so repeated smoke tests always retrain.",
        )

    if st.button(label, key=f"run_{step_id}_{cot_version}_{target}", type="primary", disabled=not allow_duplicate):
        _start_step(step_id, cot_version, fresh)

    if runs:
        st.caption("Recent runs of this step:")
        for r in runs:
            col_status, col_stop = st.columns([4, 1])
            summary = _run_config_summary(r)
            col_status.markdown(
                f"{STATUS_BADGE.get(r['status'], '⚪')} `{r['run_id']}` — {r['status']}"
                + (f"  \n<small>{summary}</small>" if summary else ""),
                unsafe_allow_html=True,
            )
            if r["status"] == "RUNNING":
                if col_stop.button("Stop", key=f"stop_{target}_{r['run_id']}"):
                    _stop_run(target, r["run_id"])
                    st.rerun()


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------

st.title("NeSy-CoT Pipeline")
st.caption(f"Knowledge graph: **{kg}** · Target: **{target}**" + (f" ({vm_profile['label']})" if vm_profile else ""))

tab_install, tab_rules, tab_cot, tab_prepare, tab_finetune, tab_monitor = st.tabs(
    ["1. Install libraries", "2. Upload rules", "3. CoT generation", "4. Prepare data", "5. Fine-tune", "Monitor & Download"]
)

with tab_install:
    st.write(
        "Local: `pip install -r requirements.txt`. "
        "VM: the CUDA 12.6 torch wheel + NF4/GPU sanity check from `README_VM.md` "
        "section 3 (run once per fresh VM)."
    )
    render_run_button("install")

with tab_rules:
    st.write(
        "Upload the cleaned AMIE rules CSV for the selected knowledge graph. "
        f"It will be written to `{run_cfg.get('kg_sparql', {}).get('rules_csv', '?')}` "
        "and picked up automatically by CoT2/CoT3 (and uploaded to the VM as part of "
        "that step, if running remotely)."
    )
    uploaded = st.file_uploader("Cleaned AMIE rules CSV", type=["csv"], key=f"rules_upload_{kg}")
    if uploaded is not None:
        try:
            df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            validate_rules_csv(df)
        except (ValueError, pd.errors.ParserError) as exc:
            st.error(f"Invalid rules CSV: {exc}")
        else:
            st.success(f"Valid: {len(df):,} rules, columns: {', '.join(df.columns)}")
            st.dataframe(df.head(20), width="stretch")
            dest = run_cfg.get("kg_sparql", {}).get("rules_csv")
            if dest and st.button("Save to project", type="primary"):
                dest_path = Path(dest)
                if not dest_path.is_absolute():
                    dest_path = PROJECT_DIR / dest_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_bytes(uploaded.getvalue())
                st.success(f"Saved to {dest_path}")

with tab_cot:
    st.subheader(STEPS["cot2"].label)
    render_run_button("cot2")
    st.divider()
    st.subheader(STEPS["cot3"].label)
    render_run_button("cot3")

with tab_prepare:
    render_run_button("prepare_data")

with tab_finetune:
    st.subheader(STEPS["step1"].label)
    render_run_button("step1")
    st.divider()
    st.subheader(STEPS["step2"].label)
    render_run_button("step2")
    st.divider()
    st.subheader(STEPS["step3"].label)
    cot_version = st.selectbox("CoT version", ["CoT2", "CoT3", "both"], key="step3_cot_version")
    render_run_button("step3", cot_version)

with tab_monitor:
    local_runs = run_store.list_local_runs()
    remote_runs = run_store.list_remote_runs()
    options = [("local", r["run_id"], r) for r in local_runs] + [("remote", r["run_id"], r) for r in remote_runs]
    options.sort(key=lambda t: t[2].get("started_at", 0), reverse=True)

    if not options:
        st.info("No runs yet — start a step from one of the tabs above.")
    else:
        def _fmt(opt) -> str:
            kind, run_id, r = opt
            return f"[{kind}] {STATUS_BADGE.get(r['status'], '⚪')} {r.get('label', r['step_id'])} — {run_id}"

        chosen = st.selectbox("Run", options, format_func=_fmt, key="monitor_run")
        kind, run_id, record = chosen

        @st.fragment(run_every="3s")
        def _live_panel(kind: str, run_id: str) -> None:
            if kind == "local":
                status = local_runner.refresh_status(run_id)
                log_tail = local_runner.tail_log(run_id)
            else:
                ssh = st.session_state.get("ssh")
                if ssh is None:
                    st.warning("Reconnect via the sidebar (SSH connection is per-session) to keep monitoring this run.")
                    status = run_store.get_remote_run(run_id)
                    log_tail = ""
                else:
                    status = remote_runner.refresh_status(ssh, run_id)
                    log_tail = remote_runner.tail_log(ssh, run_id)
            if status is None:
                st.info("Run not found.")
                return
            badge = STATUS_BADGE.get(status["status"], "⚪")
            col_title, col_stop = st.columns([5, 1])
            col_title.markdown(f"### {badge} {status.get('label', status['step_id'])} — {status['status']}")
            if status["status"] == "RUNNING" and col_stop.button("Stop", key=f"monitor_stop_{kind}_{run_id}"):
                _stop_run(kind, run_id)
                st.rerun()
            eta = remote_runner.parse_eta(log_tail)
            if eta:
                st.caption(f"Training progress: {eta}")
            if status["status"] == "FAILED":
                st.error(status.get("status_detail", "Step failed — see log below."))
            st.code(log_tail or "(no output yet)", language="text")

        _live_panel(kind, run_id)

        if kind == "remote":
            with st.expander("📂 Files on the VM (what will be archived)"):
                st.caption(
                    "Lists the current contents of ~/NeSyCoT-final/outputs and logs on "
                    "the VM. Check this before archiving — download when the run is "
                    "finished so files aren't changing mid-archive."
                )
                if st.button("List VM files", key=f"lsvm_{run_id}"):
                    ssh = st.session_state.get("ssh")
                    if ssh is None:
                        st.warning("Reconnect via the sidebar first.")
                    else:
                        with st.spinner("Listing files on the VM..."):
                            st.code(remote_runner.list_vm_files(ssh, run_id), language="text")

            col_gpu, col_archive = st.columns(2)
            with col_gpu:
                if st.button("GPU snapshot (nvidia-smi)"):
                    ssh = st.session_state.get("ssh")
                    if ssh is None:
                        st.warning("Reconnect via the sidebar first.")
                    else:
                        st.code(remote_runner.gpu_snapshot(ssh), language="text")
            with col_archive:
                if st.button("Archive & download results", type="primary"):
                    ssh = st.session_state.get("ssh")
                    if ssh is None:
                        st.warning("Reconnect via the sidebar first.")
                    else:
                        with st.spinner("Archiving on VM, downloading, and verifying checksum..."):
                            try:
                                dest_dir = run_store.RUNS_DIR / "remote" / run_id / "downloaded"
                                archive = remote_runner.archive_and_download(ssh, run_id, dest_dir)
                                st.success(f"Downloaded and verified: {archive}")
                            except SSHError as exc:
                                st.error(str(exc))
