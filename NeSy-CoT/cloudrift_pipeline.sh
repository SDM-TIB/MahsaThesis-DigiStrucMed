#!/usr/bin/env bash
set -Eeuo pipefail
RUN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$RUN_ROOT/NeSy-CoT"
mkdir -p "$RUN_ROOT/logs"
exec > >(tee -a "$RUN_ROOT/logs/pipeline.log") 2>&1
STAGE=setup
trap 'rc=$?; if (( rc != 0 )); then printf "FAILED stage=%s exit=%s\n" "$STAGE" "$rc" > "$RUN_ROOT/status.txt"; date -u; echo "FAILED stage=$STAGE exit=$rc"; fi' EXIT
stage() { STAGE="$1"; printf 'RUNNING stage=%s\n' "$STAGE" > "$RUN_ROOT/status.txt"; date -u; echo "STAGE=$STAGE"; }
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 WANDB_DISABLED=true
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=60
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
stage setup
sudo -n apt-get update
sudo -n apt-get install -y python3.12-venv tmux unzip
python3 -m venv "$RUN_ROOT/.venv"
source "$RUN_ROOT/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
cd "$PROJECT"
python -m pip install -r requirements-cloudrift-v100.txt
python -m pip check
python -m pip freeze > "$RUN_ROOT/logs/requirements.installed.txt"
nvidia-smi > "$RUN_ROOT/logs/gpu.txt"
stage gpu_check
python - <<'PY'
import torch, bitsandbytes as bnb
assert torch.cuda.is_available(), 'CUDA unavailable'
assert 'sm_70' in torch.cuda.get_arch_list(), 'V100 kernels missing'
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)
x = torch.randn(128, 128, device='cuda', dtype=torch.float16)
q, state = bnb.functional.quantize_4bit(x, quant_type='nf4')
assert torch.isfinite(bnb.functional.dequantize_4bit(q, state)).all()
print('GPU and NF4 checks passed', flush=True)
PY
stage data_audit
python -u validate_training_config.py --config T4B-config.json --baseline
stage batch_probe
python -u probe_cloudrift_batch.py
stage training_and_evaluation
python -u step3_finetune_with_rules.py --config outputs/T4B_cloudrift/config.used.json --cot_version CoT2
stage export
python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('outputs/T4B_cloudrift')
results = root / 'finetune_with_rules_CoT2_results.json'
assert results.is_file(), 'Metrics file missing'
adapter = root / 'finetuned_Qwen2.5_3B_with_rules_CoT2'
assert (adapter / 'adapter_model.safetensors').is_file(), 'Final adapter missing'
states = list(adapter.glob('checkpoint-*/trainer_state.json'))
if states:
    latest = max(states, key=lambda path: int(path.parent.name.split('-')[-1]))
    (root / 'trainer_state.final.json').write_bytes(latest.read_bytes())
files = [path for path in root.rglob('*') if path.is_file() and not any(part.startswith('checkpoint-') for part in path.parts)]
manifest = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
(root / 'result_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Final adapter, metrics and manifest verified', flush=True)
PY
cd "$RUN_ROOT"
tar --exclude='checkpoint-*' -czf results.tar.gz NeSy-CoT/outputs/T4B_cloudrift logs
sha256sum results.tar.gz > results.tar.gz.sha256
printf 'SUCCESS\n' > "$RUN_ROOT/status.txt"
date -u
echo 'SUCCESS: training, evaluation and results archive complete'
