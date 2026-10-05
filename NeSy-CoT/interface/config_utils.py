"""Build a materialized, run-specific config from the checked-in base configs.

The checked-in `guidelineKG-config.json` / `qKG-config.json` are read but
never mutated. Each run gets its own merged copy written under
`interface/runs/<local|remote>/<run_id>/config.used.json` so different runs
(different KG, VM profile, smoke/full, user edits) never collide and the
history of exactly what config produced a given result is preserved.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

BASE_CONFIGS = {
    "guidelineKG": PROJECT_DIR / "guidelineKG-config.json",
    "qKG": PROJECT_DIR / "qKG-config.json",
}


def load_base_config(kg: str) -> dict:
    path = BASE_CONFIGS[kg]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_vm_overlay(cfg: dict, vm_profile: dict | None) -> dict:
    """Only gpu_max_memory_mb varies by VM profile — see plan for rationale."""
    cfg = copy.deepcopy(cfg)
    if vm_profile is not None:
        cfg["gpu_max_memory_mb"] = vm_profile["gpu_max_memory_mb"]
    return cfg


_CANONICAL_PREGENERATED_FILES = {
    "train_without_rules_csv": "train_data_without_rules.csv",
    "train_with_rules_cot2_csv": "train_data_with_rules_CoT2.csv",
    "train_with_rules_cot3_csv": "train_data_with_rules_CoT3.csv",
    "test_eval_clean_csv": "test_eval_clean.csv",
    "test_shared_cot2_csv": "test_shared_CoT2.csv",
    "test_shared_cot3_csv": "test_shared_CoT3.csv",
    "test_shared_baseline_csv": "test_shared_baseline.csv",
}


def apply_smoke_transform(cfg: dict) -> dict:
    """Shrink training/eval to a fast sanity-check run.

    Only num_steps/logging/eval-sample-count and output_dir (where the
    smoke run's own results/checkpoints land, so they never overwrite a
    real run's) change. Input data paths must still resolve against the
    *original* output_dir, since prepare_data.py is not re-run in smoke
    mode — so every canonical pregenerated_data key is pinned to the
    original output_dir up front, before output_dir itself is changed.
    Leaving this to each step's own output_dir-relative fallback (as
    step1/2/3_*.py do) would silently point unset keys at the now-empty
    smoke directory instead.
    """
    cfg = copy.deepcopy(cfg)
    base_output_dir = Path(cfg.get("output_dir", "./outputs/"))

    pregen = cfg.setdefault("pregenerated_data", {})
    pregen.setdefault("use_pregenerated", True)
    for key, filename in _CANONICAL_PREGENERATED_FILES.items():
        pregen.setdefault(key, str(base_output_dir / filename))

    tcfg = cfg.setdefault("training", {})
    tcfg["num_steps"] = 10
    tcfg["warmup_steps"] = 0
    tcfg["logging_steps"] = 1
    tcfg["save_steps"] = 1000
    ecfg = cfg.setdefault("evaluation", {})
    ecfg["max_samples"] = min(ecfg.get("max_samples", 20), 20)

    cfg["output_dir"] = str(base_output_dir).rstrip("/\\") + "_smoke/"
    return cfg


def deep_merge(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def build_run_config(
    kg: str,
    vm_profile: dict | None = None,
    smoke: bool = False,
    model_key: str | None = None,
    user_edits: dict | None = None,
) -> dict:
    cfg = load_base_config(kg)
    if model_key:
        cfg["model_key"] = model_key
    cfg = apply_vm_overlay(cfg, vm_profile)
    if smoke:
        cfg = apply_smoke_transform(cfg)
    if user_edits:
        cfg = deep_merge(cfg, user_edits)
    # Never persist a real token to disk — callers set the HF_TOKEN env var
    # for the process/session instead (see utils.load_config's resolution
    # order: env var takes priority over this field).
    cfg["huggingface_token"] = None
    return cfg


def materialize_config(cfg: dict, dest_path: Path) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return dest_path
