"""Declarative registry of NeSy-CoT pipeline steps.

Single source of truth for: what script runs a step, what extra CLI args it
needs, and — for VM runs — exactly which local files must be staged on the
remote host before the step can run. Both local_runner.py and
remote_runner.py import this instead of re-deriving command lines, so the
two execution paths can't silently drift apart.

Remote install/GPU-check commands are copied verbatim from the
already-debugged sequence in README_VM.md section 3 (wrong CUDA wheel and
missing sm_70 kernels were real, previously-hit failures on this exact VM
family) rather than re-derived.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Directory name (under the VM's $HOME — resolved to an absolute path by
# remote_runner.resolve_root, since SFTP does not expand '~') matching the
# directory name already used in README_VM.md, so a manual session and an
# interface-driven run share the same venv/uploaded files.
REMOTE_ROOT_NAME = "NeSyCoT-final"


@dataclass(frozen=True)
class Step:
    id: str
    label: str
    script: Optional[str]  # None for the pseudo-steps (install, upload_rules)
    needs_gpu: bool
    code_files: tuple[str, ...] = ()  # extra .py files (besides `script`) the step imports
    build_args: Optional[Callable[[str, Optional[str]], list[str]]] = None
    data_paths: Optional[Callable[[dict, Optional[str]], list[Path]]] = None


def _cfg_args(cfg_path: str, _cot_version: Optional[str]) -> list[str]:
    return ["--config", cfg_path]


def _step3_args(cfg_path: str, cot_version: Optional[str]) -> list[str]:
    args = ["--config", cfg_path]
    if cot_version:
        args += ["--cot_version", cot_version]
    return args


def _cot_data(cfg: dict, _cot_version: Optional[str]) -> list[Path]:
    sparql = cfg.get("kg_sparql", {})
    return [Path(sparql[key]) for key in ("rules_csv", "kg_file") if sparql.get(key)]


def _prepare_data_data(cfg: dict, _cot_version: Optional[str]) -> list[Path]:
    paths = [Path(cfg["data_dir"])]
    for key in ("rules_dir_cot2", "rules_dir_cot3"):
        val = cfg.get(key)
        if val and Path(val).exists():
            paths.append(Path(val))
    return paths


def _resolve_pregenerated(cfg: dict, *keyed_fallbacks: tuple[str, str]) -> list[Path]:
    """Resolve pregenerated_data CSV paths, falling back to output_dir/<name>
    the same way step1/2/3_*.py and resolve_shared_test_path() do."""
    pregen = cfg.get("pregenerated_data", {})
    output_dir = Path(cfg.get("output_dir", "./outputs/"))
    return [Path(pregen.get(key) or (output_dir / fallback)) for key, fallback in keyed_fallbacks]


def _step1_data(cfg: dict, _cot_version: Optional[str]) -> list[Path]:
    return _resolve_pregenerated(cfg, ("test_eval_clean_csv", "test_eval_clean.csv"))


def _step2_data(cfg: dict, _cot_version: Optional[str]) -> list[Path]:
    return _resolve_pregenerated(
        cfg,
        ("train_without_rules_csv", "train_data_without_rules.csv"),
        ("test_eval_clean_csv", "test_eval_clean.csv"),
    )


def _step3_data(cfg: dict, cot_version: Optional[str]) -> list[Path]:
    versions = ["CoT2", "CoT3"] if cot_version in (None, "both") else [cot_version]
    keyed = []
    for v in versions:
        vl = v.lower()
        keyed.append((f"train_with_rules_{vl}_csv", f"train_data_with_rules_{v}.csv"))
        keyed.append((f"test_shared_{vl}_csv", f"test_shared_{v}.csv"))
    keyed.append(("test_eval_clean_csv", "test_eval_clean.csv"))
    return _resolve_pregenerated(cfg, *keyed)


STEPS: dict[str, Step] = {
    "install": Step("install", "1. Install libraries", script=None, needs_gpu=False),
    "upload_rules": Step("upload_rules", "2. Upload rules (AMIE, cleaned)", script=None, needs_gpu=False),
    "cot2": Step(
        "cot2", "3a. Run CoT2", script="NL-instances-CoT2.py", needs_gpu=False,
        build_args=_cfg_args, data_paths=_cot_data,
    ),
    "cot3": Step(
        "cot3", "3b. Run CoT3 (optional)", script="NL-instances-CoT3.py", needs_gpu=False,
        build_args=_cfg_args, data_paths=_cot_data,
    ),
    "prepare_data": Step(
        "prepare_data", "4. Prepare data", script="prepare_data.py", needs_gpu=False,
        code_files=("utils.py", "dataset_io.py"),
        build_args=_cfg_args, data_paths=_prepare_data_data,
    ),
    "step1": Step(
        "step1", "5a. Fine-tune Step 1 — evaluate base model",
        script="step1_evaluate_base.py", needs_gpu=True,
        code_files=("utils.py",), build_args=_cfg_args, data_paths=_step1_data,
    ),
    "step2": Step(
        "step2", "5b. Fine-tune Step 2 — without rules",
        script="step2_finetune_no_rules.py", needs_gpu=True,
        code_files=("utils.py",), build_args=_cfg_args, data_paths=_step2_data,
    ),
    "step3": Step(
        "step3", "5c. Fine-tune Step 3 — with rules",
        script="step3_finetune_with_rules.py", needs_gpu=True,
        code_files=("utils.py",), build_args=_step3_args, data_paths=_step3_data,
    ),
}

ORDERED_STEP_IDS = ["install", "upload_rules", "cot2", "cot3", "prepare_data", "step1", "step2", "step3"]


def adapter_output_dirs(cfg: dict, step_id: str, cot_version: Optional[str]) -> list[Path]:
    """Directories a training step writes its LoRA adapter + checkpoints into.

    Matches the names step2/step3_*.py build:
      step2 -> finetuned_<tag>_baseline
      step3 -> finetuned_<tag>_with_rules_<CoTx>
    Returned so a "fresh start" can clear them first — otherwise
    utils.fine_tune_model auto-resumes from a leftover checkpoint
    (get_last_checkpoint), which for a re-run means it skips training and
    goes straight to eval. Empty for non-training steps (step1 has no adapter).
    """
    output_dir = Path(cfg["output_dir"])
    tag = cfg["model_key"].replace("-", "_")
    if step_id == "step2":
        return [output_dir / f"finetuned_{tag}_baseline"]
    if step_id == "step3":
        versions = ["CoT2", "CoT3"] if cot_version in (None, "both") else [cot_version]
        return [output_dir / f"finetuned_{tag}_with_rules_{v}" for v in versions]
    return []


def install_command_local() -> str:
    return f'"{sys.executable}" -m pip install -r requirements.txt'


# Verbatim from README_VM.md section 3 — CUDA 12.6 torch wheel + NF4/GPU
# sanity check. Do not "simplify" the pinned torch version or drop the
# sm_70 check: both were added after real failures on this VM family.
# '__ROOT__' is substituted with the resolved absolute remote path by
# remote_runner.build_run_script (SFTP doesn't expand '~', so the script
# text and the SFTP-uploaded files must agree on one absolute directory).
# Assumes passwordless sudo, standard on CloudRift/most GPU rental images —
# if that's not the case here, this step's log will show the sudo prompt
# hanging/failing rather than silently skipping the apt install.
REMOTE_INSTALL_SCRIPT = r"""set -Eeuo pipefail
cd __ROOT__
sudo apt update
sudo apt install -y python3.12-venv python3.12-dev build-essential tmux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade "torch==2.14.0+cu126" --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
python - <<'PY'
import torch
import bitsandbytes as bnb
assert torch.cuda.is_available(), 'CUDA unavailable'
assert 'sm_70' in torch.cuda.get_arch_list(), 'V100 kernels missing'
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
x = torch.randn(128, 128, device='cuda', dtype=torch.float16)
q, state = bnb.functional.quantize_4bit(x, quant_type='nf4')
y = bnb.functional.dequantize_4bit(q, state)
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print('GPU and 4-bit checks passed')
PY
"""
