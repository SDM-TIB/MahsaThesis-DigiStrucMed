# CloudRift V100: run from scratch and save results

VM: `riftuser@66.172.10.137`, Ubuntu 24.04, V100 SXM2 (16 GB), 6 CPUs, 52 GB RAM. Confirm the IP in CloudRift before starting; replace it throughout if it changes. Use your CloudRift SSH password/key. Commands assume the default port.

## Which terminal?

| Terminal | Prompt | Commands |
| --- | --- | --- |
| **LOCAL POWERSHELL** | `PS C:\Users\...>` | Upload, connect, download |
| **REMOTE SSH** | `riftuser@riftvm:~$` | Install, train, monitor, archive |

After `ssh`, the same window runs remote commands until you type `exit`. A `(.venv)` prefix can appear on either computer: check the rest of the prompt. Copy commands only, not prompt examples.

## 1. LOCAL POWERSHELL: upload all required files

Open a new local PowerShell window:

```powershell
# LOCAL POWERSHELL (prompt: PS C:\...\NeSyCoT-final>) — not the SSH session
cd "C:\Users\tehranim.TIB.001\Desktop\Computer-20260504T112359Z-3-001\Computer\Master Thesis\NeSyCoT\NeSyCoT-final"
ssh riftuser@66.172.10.137 "mkdir -p ~/NeSyCoT-final/outputs/qKG"
scp config.json utils.py step1_evaluate_base.py step2_finetune_no_rules.py step3_finetune_with_rules.py riftuser@66.172.10.137:~/NeSyCoT-final/
scp outputs/qKG/test_eval_clean.csv outputs/qKG/train_data_without_rules.csv outputs/qKG/train_data_with_rules_CoT2.csv outputs/qKG/test_shared_CoT2.csv riftuser@66.172.10.137:~/NeSyCoT-final/outputs/qKG/
```

Wait for each command to succeed. This uploads all nine required files: five configuration/code files and four prepared CSVs. KG folders, raw rules, ZIPs and preparation scripts are not needed. Step 3 uses CoT2 because the local CoT3 CSVs are missing.

## 2. LOCAL POWERSHELL: connect

```powershell
# LOCAL POWERSHELL (prompt: PS C:\...\NeSyCoT-final>)
ssh riftuser@66.172.10.137
```

Wait for the prompt to change to `riftuser@riftvm:~$` — that confirms you are now on the VM, not local PowerShell. Every block below in Sections 3-7 runs there, unless the block is explicitly marked LOCAL POWERSHELL again.

## 3. REMOTE SSH: install the environment

Run each block successfully before continuing:

```bash
# REMOTE SSH (prompt: riftuser@riftvm:~$) — not local PowerShell
cd ~/NeSyCoT-final
nvidia-smi
sudo apt update
sudo apt install -y python3.12-venv python3.12-dev build-essential tmux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install CUDA 12.6 PyTorch first. The CUDA 13 build previously installed on this VM lacked V100 kernels. The CUDA version in `nvidia-smi` describes driver capability, not the required PyTorch wheel. See the [official wheel index](https://download.pytorch.org/whl/cu126/torch/).

**REMOTE SSH** (prompt: `riftuser@riftvm:~$`):

```bash
python -m pip install --upgrade "torch==2.14.0+cu126" --index-url https://download.pytorch.org/whl/cu126
python -m pip install "torch==2.14.0+cu126" bitsandbytes transformers peft accelerate datasets sentencepiece scikit-learn pandas numpy huggingface_hub
python -m pip check
```

**REMOTE SSH** (prompt: `riftuser@riftvm:~$`) — verify GPU computation and NF4 before starting Step 1:

```bash
python - <<'PY'
import torch
import bitsandbytes as bnb
import utils
assert utils._ML_AVAILABLE, 'Missing ML dependency'
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
```

If this fails, fix the environment before proceeding. The config targets Llama 3.2 1B, NF4, FP16, batch size 1, gradient accumulation 4, 14,000 MB model-placement memory, 1,024 training/input tokens and 256 generated tokens. Check `config.json` for current training-step and evaluation-sample limits.

## 4. REMOTE SSH: start a persistent terminal

```bash
# REMOTE SSH (prompt: riftuser@riftvm:~$)
tmux new -s nesycot
```

Inside this **REMOTE SSH tmux session** (prompt still `riftuser@riftvm:~$`, now inside `tmux`):

```bash
cd ~/NeSyCoT-final
source .venv/bin/activate
read -rsp 'Hugging Face token: ' HF_TOKEN
echo
export HF_TOKEN
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
```

Enter a token with access to the configured Llama repository at the hidden prompt.

## 5. REMOTE SSH inside tmux: run all three steps and save logs

Paste the entire block below. It runs each step only after the previous one succeeds. It saves output and errors in a timestamped folder and stops on failure.

```bash
# REMOTE SSH, inside the tmux session (prompt: riftuser@riftvm:~$)
bash <<'RUN'
set -Eeuo pipefail
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
export RUN_DIR="$PWD/logs/$RUN_ID"
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > logs/latest_run.txt
exec > >(tee -a "$RUN_DIR/pipeline.log") 2>&1
trap 'rc=$?; echo "FAILED at line $LINENO, exit $rc"; echo "FAILED exit=$rc" > "$RUN_DIR/status.txt"; exit "$rc"' ERR
echo RUNNING > "$RUN_DIR/status.txt"
python -m pip freeze > "$RUN_DIR/environment.txt"
nvidia-smi > "$RUN_DIR/gpu.txt"
cp utils.py step1_evaluate_base.py step2_finetune_no_rules.py step3_finetune_with_rules.py "$RUN_DIR/"
python - <<'PY'
import json, os
from pathlib import Path
import pandas as pd
cfg = json.loads(Path('config.json').read_text())
assert Path(cfg['output_dir']).resolve() == Path('outputs/qKG').resolve()
for name in ['test_eval_clean.csv', 'train_data_without_rules.csv',
             'train_data_with_rules_CoT2.csv', 'test_shared_CoT2.csv']:
    df = pd.read_csv(Path('outputs/qKG') / name)
    assert len(df) and {'Prompt', 'Label'} <= set(df.columns), name
    print(name, len(df), 'rows')
cfg['huggingface_token'] = 'REDACTED: supplied via HF_TOKEN'
(Path(os.environ['RUN_DIR']) / 'config.redacted.json').write_text(json.dumps(cfg, indent=2))
PY
sha256sum outputs/qKG/test_eval_clean.csv outputs/qKG/train_data_without_rules.csv outputs/qKG/train_data_with_rules_CoT2.csv outputs/qKG/test_shared_CoT2.csv > "$RUN_DIR/inputs.sha256"

echo 'STEP 1/3: evaluate base model'
python -u step1_evaluate_base.py --config config.json 2>&1 | tee "$RUN_DIR/step1.log"

echo 'STEP 2/3: fine-tune without rules and evaluate'
python -u step2_finetune_no_rules.py --config config.json 2>&1 | tee "$RUN_DIR/step2.log"

echo 'STEP 3/3: fine-tune with CoT2 rules and evaluate'
python -u step3_finetune_with_rules.py --config config.json --cot_version CoT2 2>&1 | tee "$RUN_DIR/step3.log"

python - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path('config.json').read_text())
out = Path(cfg['output_dir'])
tag = cfg['model_key'].replace('-', '_')
for name in ['base_model_results.json', 'finetune_no_rules_results.json',
             'finetune_with_rules_CoT2_results.json']:
    data = json.loads((out / name).read_text())
    assert data['model'] == cfg['model_key'], name
    print(name, data['metrics'])
for suffix in ['baseline', 'with_rules_CoT2']:
    folder = out / f'finetuned_{tag}_{suffix}'
    assert (folder / 'adapter_config.json').is_file(), str(folder)
    assert any((folder / n).is_file() for n in
               ['adapter_model.safetensors', 'adapter_model.bin']), str(folder)
PY
echo SUCCESS > "$RUN_DIR/status.txt"
echo 'All three steps completed and results verified.'
date -u
RUN
```

Logs are saved automatically under `logs/TIMESTAMP/`:

- `pipeline.log`: combined output.
- `step1.log`, `step2.log`, `step3.log`: individual step output and errors.
- `status.txt`: RUNNING, FAILED, or SUCCESS.
- Environment, GPU, redacted configuration, input checksums, and copies of the Python scripts.

The printed input row counts may exceed the configured evaluation sample limit. Run only one experiment at a time. This starts training from scratch and does not automatically resume checkpoints. Reruns create new logs but overwrite fixed result/adapter paths; archive previous results first.

Detach with **Ctrl+B, release, then D**. The job continues after SSH disconnects. Do not press Ctrl+C in the training pane: that interrupts training.

## 6. Monitor from a second terminal

**LOCAL POWERSHELL: open a *second*, brand-new window and connect** (prompt: `PS C:\...>`):

```powershell
ssh riftuser@66.172.10.137
```

**Second REMOTE SSH session** (prompt: `riftuser@riftvm:~$`, a different window from the tmux training session):

```bash
cd ~/NeSyCoT-final
RUN_DIR=$(cat logs/latest_run.txt)
cat "$RUN_DIR/status.txt"
tail -n 50 -F "$RUN_DIR/pipeline.log"
```

Ctrl+C here stops tail only. To monitor one step, use `tail -n 50 -F "$RUN_DIR/step2.log"` or `step3.log`; each file appears when its step begins.

**REMOTE SSH** (same second session, prompt: `riftuser@riftvm:~$`) — monitor GPU usage (Ctrl+C stops this monitor):

```bash
watch -n 5 nvidia-smi
```

**REMOTE SSH: return to the original training terminal** (still the *first* SSH window, prompt: `riftuser@riftvm:~$`):

```bash
tmux attach -t nesycot
```

A tmux session existing does not prove the job is running. Check status and the end of the log. For FAILED status, resolve the logged error before retrying. A stopped or terminated VM cannot keep the job running.

## 7. REMOTE SSH: archive results and logs

After SUCCESS, the results have already been saved automatically:

```text
outputs/qKG/
  base_model_results.json
  finetune_no_rules_results.json
  finetune_with_rules_CoT2_results.json
  finetuned_LLaMA_3.2_1B_baseline/
  finetuned_LLaMA_3.2_1B_with_rules_CoT2/
```

The model folders contain LoRA adapters and checkpoints. Reload them with the same base model; they are not standalone merged models.

**REMOTE SSH** (prompt: `riftuser@riftvm:~$` — the archive step needs the actual shell, not PowerShell: `tar`, `sha256sum`, and `&&` used below are bash syntax and will error in PowerShell):

```bash
cd ~/NeSyCoT-final
RUN_DIR=$(cat logs/latest_run.txt)
cat "$RUN_DIR/status.txt"
mkdir -p final_outputs
ARCHIVE="final_outputs/nesycot_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
tar -czf "$ARCHIVE" outputs/qKG logs && sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
ls -lh "$ARCHIVE" "$ARCHIVE.sha256"
```

Archive after training stops so files do not change during the copy. This saves metrics, adapters, checkpoints, input CSVs and all logs, including the redacted run config. Failed runs can also be archived for diagnosis but contain partial results. Copy the exact filename the `ls` line prints (e.g. `nesycot_20260908T191234Z.tar.gz`) — you need it verbatim in Section 8.

## 8. LOCAL POWERSHELL: download and verify

Open a new **local** PowerShell window (prompt: `PS C:\...>`, not the SSH session). These commands read `latest_archive.txt` from the VM, so no timestamp needs to be typed manually:

```powershell
# LOCAL POWERSHELL (prompt: PS C:\...\NeSyCoT-final>)
$vmAddress = "riftuser@66.172.10.137"
$remoteArchive = (ssh $vmAddress "cat /home/riftuser/NeSyCoT-final/final_outputs/latest_archive.txt").Trim()
if (-not $remoteArchive) { throw "The VM did not return an archive path. Run Section 7 first." }
$archiveName = Split-Path $remoteArchive -Leaf
Write-Host "Downloading $archiveName"
New-Item -ItemType Directory -Force .\cloudrift-results | Out-Null
scp "${vmAddress}:/home/riftuser/NeSyCoT-final/$remoteArchive" .\cloudrift-results\
scp "${vmAddress}:/home/riftuser/NeSyCoT-final/$remoteArchive.sha256" .\cloudrift-results\
$expectedHash = ((Get-Content ".\cloudrift-results\$archiveName.sha256") -split '\s+')[0]
$actualHash = (Get-FileHash ".\cloudrift-results\$archiveName" -Algorithm SHA256).Hash
if ($actualHash -ne $expectedHash) { throw "Checksum mismatch: download again" }
Write-Host "Download verified"
tar -xzf ".\cloudrift-results\$archiveName" -C .\cloudrift-results
```

Extracted results are under `cloudrift-results/outputs/qKG/`; logs are under `cloudrift-results/logs/`. Verify the download before terminating the VM.

## Troubleshooting

- **Missing ensurepip:** in REMOTE SSH install `python3.12-venv`, recreate and activate the environment.
- **Missing Python.h/compiler:** in REMOTE SSH install `python3.12-dev build-essential`.
- **No kernel image / missing sm_70:** install the exact CUDA 12.6 wheel in Section 3 and rerun the GPU checks. For ops.cu errors, check NF4 too and inspect any remaining error.
- **SCP cannot find config.json:** run SCP in LOCAL POWERSHELL from the project folder.
- **CUDA unavailable with a PS C:\Users prompt:** Python is running locally; connect to the VM.
- **`The token '&&' is not a valid statement separator` / bash commands erroring in PowerShell:** you pasted a REMOTE SSH block into a local PowerShell window. Check the prompt: `riftuser@riftvm:~$` means you're on the VM, `PS C:\...>` means you're local. SSH in first, then paste bash blocks.
- **Archive not found:** run Section 7 on the VM first and confirm that `final_outputs/latest_archive.txt` names an existing file. Section 8 retrieves the name automatically.
- **401/403:** verify model access and re-enter the token at the hidden prompt inside tmux.
- **OOM:** check for other GPU processes; record configuration changes and check token coverage before reducing input limits.
