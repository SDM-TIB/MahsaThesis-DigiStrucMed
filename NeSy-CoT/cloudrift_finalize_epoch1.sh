#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$RUN_ROOT/NeSy-CoT"
OUTPUT="$PROJECT/outputs/T4B_cloudrift"
ADAPTER="$OUTPUT/finetuned_Qwen2.5_3B_with_rules_CoT2"
CHECKPOINT="$ADAPTER/checkpoint-2855"

mkdir -p "$RUN_ROOT/logs"
exec > >(tee -a "$RUN_ROOT/logs/finalize.log") 2>&1

STAGE=initializing_epoch1_finalize
trap 'rc=$?; if (( rc != 0 )); then printf "FAILED stage=%s exit=%s\n" "$STAGE" "$rc" > "$RUN_ROOT/status.txt"; date -u; echo "FAILED stage=$STAGE exit=$rc"; fi' EXIT
stage() {
    STAGE="$1"
    printf 'RUNNING stage=%s\n' "$STAGE" > "$RUN_ROOT/status.txt"
    date -u
    echo "STAGE=$STAGE"
}

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 WANDB_DISABLED=true
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=60
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

test -f "$CHECKPOINT/adapter_model.safetensors"
test -f "$CHECKPOINT/trainer_state.json"

source "$RUN_ROOT/.venv/bin/activate"
cd "$PROJECT"

stage epoch1_checkpoint_evaluation
python -u step3_finetune_with_rules.py \
    --config outputs/T4B_cloudrift/config.epoch1.used.json \
    --cot_version CoT2

stage export
python - <<'PY'
import hashlib
import json
import shutil
from pathlib import Path

root = Path('outputs/T4B_cloudrift')
adapter = root / 'finetuned_Qwen2.5_3B_with_rules_CoT2'
checkpoint = adapter / 'checkpoint-2855'
results = root / 'finetune_with_rules_CoT2_results.json'

assert results.is_file(), 'Metrics file missing'
assert (adapter / 'adapter_model.safetensors').is_file(), 'Final adapter missing'
assert (checkpoint / 'trainer_state.json').is_file(), 'Epoch-1 trainer state missing'
shutil.copy2(checkpoint / 'trainer_state.json', root / 'trainer_state.final.json')

early_stop = {
    'selected_checkpoint': str(checkpoint),
    'completed_epochs': 1,
    'global_step': 2855,
    'validation_loss': 0.00014080021355766803,
    'reason': 'Training and validation loss converged after one full pass; later training loss was zero with near-zero gradients.'
}
(root / 'early_stop.json').write_text(json.dumps(early_stop, indent=2) + '\n', encoding='utf-8')

files = [
    path for path in root.rglob('*')
    if path.is_file()
    and path.name != 'result_manifest.json'
    and not any(part.startswith('checkpoint-') for part in path.parts)
]
manifest = {
    str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in files
}
(root / 'result_manifest.json').write_text(
    json.dumps(manifest, indent=2) + '\n', encoding='utf-8'
)
print('Epoch-1 adapter, metrics and manifest verified', flush=True)
PY

cd "$RUN_ROOT"
tar --exclude='checkpoint-*' -czf results.tar.gz NeSy-CoT/outputs/T4B_cloudrift logs
sha256sum results.tar.gz > results.tar.gz.sha256
printf 'SUCCESS\n' > "$RUN_ROOT/status.txt"
date -u
echo 'SUCCESS: epoch-1 checkpoint evaluated and packaged'
