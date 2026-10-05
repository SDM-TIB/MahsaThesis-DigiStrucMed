# Run the 3 fine-tuning steps for guidelineKG on a CloudRift V100 (from scratch)

This is the **manual, copy-paste** guide for fine-tuning **Llama-3.2-1B** on
**guidelineKG** on a rented GPU VM, running all three steps:

1. **Step 1** — evaluate the base model (no fine-tuning)
2. **Step 2** — fine-tune WITHOUT rules
3. **Step 3** — fine-tune WITH rules (CoT2)

> **Easier alternative:** the Streamlit app in [interface/](interface/) does
> every step below with buttons (upload, run, monitor, download). See
> [README.md](README.md). Use this manual guide if you prefer the raw commands
> or need to debug.

**Target VM:** Tesla **V100-SXM2-16GB**, Ubuntu 24.04, `riftuser@<VM_IP>`.
Replace `<VM_IP>` everywhere with your instance's current IP from the CloudRift
dashboard (it changes every time you rent a new instance). The config is tuned
for this **16 GB** card (`gpu_max_memory_mb: 14000`, batch 1 / grad-accum 4 —
the memory-safe settings proven on this exact card). If you instead rent the
**32 GB SXM3**, you can raise `gpu_max_memory_mb` to `28000` and set batch 4 /
grad-accum 1 in `guidelineKG-config.vm.json` for faster training.

**Rough time / cost:** at $0.28/hr, install ≈ 10–15 min, the 3 steps ≈ 1–2 h.
Budget under ~$1. The 640 MB training CSV upload dominates the transfer time.

---

## Which terminal am I in?

| Terminal | Prompt looks like | Used for |
| --- | --- | --- |
| **LOCAL POWERSHELL** | `PS C:\...>` | upload, connect, download |
| **REMOTE SSH** | `riftuser@...:~$` | install, train, monitor, archive |

After `ssh`, the same window becomes the VM until you type `exit`. Always check
the prompt before pasting: bash blocks pasted into PowerShell (or vice-versa)
will error. Copy the commands, not the `# LABEL` comment lines.

---

## 0. Prerequisites (do these once, BEFORE renting the VM)

### 0a. Get Llama-3.2 access on Hugging Face  ← the most common failure

Llama is a **gated** model. If your HF account is not approved, Step 1 fails
partway with `GatedRepoError: 403 ... Access to model
meta-llama/Llama-3.2-1B-Instruct is restricted`.

1. Log in to Hugging Face with the account whose token you will use.
2. Open <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct> and accept
   Meta's license / fill the access form. Approval for 3.2 is usually quick.
3. Use a **Read** token from that same account (Settings → Access Tokens). A
   fine-grained token also needs *"Read access to contents of all public gated
   repos you can access"*.

You will paste this token later at a hidden prompt (Section 4) — never put it
in a file or in the config.

### 0b. Confirm the local files exist

**LOCAL POWERSHELL:**

```powershell
cd "C:\Users\tehranim.TIB.001\Desktop\Computer-20260504T112359Z-3-001\Computer\Master Thesis\MahsaThesis-DigiStrucMed\NeSy-CoT"
# code + config
ls requirements.txt, utils.py, guidelineKG-config.vm.json, step1_evaluate_base.py, step2_finetune_no_rules.py, step3_finetune_with_rules.py
# the 4 data CSVs the 3 steps read
ls outputs\guidelineKG\test_eval_clean.csv, outputs\guidelineKG\train_data_without_rules.csv, outputs\guidelineKG\train_data_with_rules_CoT2.csv, outputs\guidelineKG\test_shared_CoT2.csv
```

If any CSV is missing, run `python prepare_data.py --config guidelineKG-config.json`
first (see [README.md](README.md)) — the VM only receives already-prepared data.

`guidelineKG-config.vm.json` is pre-set for this run: **Llama-3.2-1B**, batch
size 1 / grad-accum 4 (effective batch 4 — memory-safe on the 16 GB card),
`gpu_max_memory_mb: 14000`, `num_steps: 2000`, `max_length: 1024`,
`max_new_tokens: 256`. Token budget stays generous so the label at the end of
each sequence is never truncated.

---

## 1. LOCAL POWERSHELL: upload the required files

Rent the VM, note its IP, then:

```powershell
# LOCAL POWERSHELL (prompt: PS C:\...\NeSy-CoT>)
cd "C:\Users\tehranim.TIB.001\Desktop\Computer-20260504T112359Z-3-001\Computer\Master Thesis\MahsaThesis-DigiStrucMed\NeSy-CoT"
$vm = "riftuser@185.165.50.63"

ssh $vm "mkdir -p ~/NeSyCoT-final/outputs/guidelineKG"

# code + config (small, fast). -o ServerAliveInterval=15 keeps the link alive.
scp -o ServerAliveInterval=15 requirements.txt utils.py guidelineKG-config.vm.json step1_evaluate_base.py step2_finetune_no_rules.py step3_finetune_with_rules.py "${vm}:~/NeSyCoT-final/"

# data: the 4 CSVs are ~900 MB raw (train_data_with_rules_CoT2.csv alone is 640 MB).
# Pushing that in one multi-file scp often trips "Connection reset" on budget GPU
# hosts. Instead gzip them into ONE file (text shrinks ~5x -> ~180 MB), send that,
# and unpack on the VM — one small transfer is far more reliable.
tar -czf guidelineKG_data.tar.gz -C outputs/guidelineKG test_eval_clean.csv train_data_without_rules.csv train_data_with_rules_CoT2.csv test_shared_CoT2.csv
scp -o ServerAliveInterval=15 -o ServerAliveCountMax=8 guidelineKG_data.tar.gz "${vm}:~/NeSyCoT-final/"
ssh $vm "cd ~/NeSyCoT-final && mkdir -p outputs/guidelineKG && tar -xzf guidelineKG_data.tar.gz -C outputs/guidelineKG && rm guidelineKG_data.tar.gz && ls -lh outputs/guidelineKG"
Remove-Item guidelineKG_data.tar.gz
```

The final `ls -lh` should list all four CSVs on the VM. That is **everything**
the 3 steps need — KG folders, raw rules, ZIPs and the prepare scripts are not
uploaded.

---

## 2. LOCAL POWERSHELL: connect to the VM

```powershell
ssh riftuser@<VM_IP>
```

Wait for the prompt to change to `riftuser@...:~$`. Everything in Sections 3–7
runs there, except where a block is explicitly re-labelled LOCAL POWERSHELL.

---

## 3. REMOTE SSH: install the environment (once per VM)

```bash
# REMOTE SSH (prompt: riftuser@...:~$)
cd ~/NeSyCoT-final
nvidia-smi
sudo apt update
sudo apt install -y python3.12-venv python3.12-dev build-essential tmux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the **CUDA 12.6** PyTorch wheel first (a CUDA 13 build lacks V100
`sm_70` kernels — the driver's CUDA version in `nvidia-smi` is not the wheel you
need). See the [wheel index](https://download.pytorch.org/whl/cu126/torch/).

```bash
# REMOTE SSH
python -m pip install --upgrade "torch==2.14.0+cu126" --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
```

Verify the GPU and 4-bit path before spending time on a real step:

```bash
# REMOTE SSH
python - <<'PY'
import torch, bitsandbytes as bnb
assert torch.cuda.is_available(), 'CUDA unavailable'
assert 'sm_70' in torch.cuda.get_arch_list(), 'V100 kernels missing'
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
x = torch.randn(128, 128, device='cuda', dtype=torch.float16)
q, s = bnb.functional.quantize_4bit(x, quant_type='nf4')
assert torch.isfinite(bnb.functional.dequantize_4bit(q, s)).all()
print('GPU and 4-bit checks passed')
PY
```

If this fails, fix the environment before continuing.

---

## 4. REMOTE SSH: start a persistent session and set the token

`tmux` keeps the job running if SSH drops or you close the laptop.

```bash
# REMOTE SSH
tmux new -s nesycot
```

Inside the tmux session:

```bash
# REMOTE SSH, inside tmux
cd ~/NeSyCoT-final
source .venv/bin/activate
read -rsp 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
```

Paste the token from Section 0a at the hidden prompt (nothing shows as you
type). It stays in memory only.

**Confirm the token is valid AND has Llama access before running** — this
catches both the `401 invalid token` and the `403 no access` failures in
seconds, instead of after the model starts loading:

```bash
# REMOTE SSH, inside tmux
echo "token length=${#HF_TOKEN}  starts=${HF_TOKEN:0:3}"   # expect ~37 chars, starts 'hf_'
huggingface-cli whoami                                     # prints your username if the token is valid
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct config.json --quiet && echo "ACCESS OK"
```

- `whoami` fails with **401 / Invalid token** → the token string is wrong or
  expired. Copy a fresh **Read** token from
  <https://huggingface.co/settings/tokens> and redo the `read`/`export` above.
  (`HF_TOKEN` overrides `huggingface-cli login`, so you must fix the env var
  itself — logging in separately won't help.)
- Download fails with **403** → token is valid but Section 0a (Llama access) is
  not approved yet.
- `ACCESS OK` → you're ready to run Section 5.

---

## 5. REMOTE SSH inside tmux: run all 3 steps with logging

Paste the whole block. It runs each step only if the previous one succeeds,
saves per-step logs, and writes `status.txt` (RUNNING / FAILED / SUCCESS).

```bash
# REMOTE SSH, inside tmux
bash <<'RUN'
set -Eeuo pipefail
cd ~/NeSyCoT-final
source .venv/bin/activate
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
export RUN_DIR="$PWD/logs/$RUN_ID"
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > logs/latest_run.txt
exec > >(tee -a "$RUN_DIR/pipeline.log") 2>&1
trap 'rc=$?; echo "FAILED at line $LINENO (exit $rc)"; echo "FAILED exit=$rc" > "$RUN_DIR/status.txt"; exit "$rc"' ERR
echo RUNNING > "$RUN_DIR/status.txt"
export WANDB_DISABLED=true PYTHONUNBUFFERED=1
CFG=guidelineKG-config.vm.json

echo "===== STEP 1/3: evaluate base model ====="
python -u step1_evaluate_base.py --config "$CFG" 2>&1 | tee "$RUN_DIR/step1.log"

echo "===== STEP 2/3: fine-tune WITHOUT rules ====="
python -u step2_finetune_no_rules.py --config "$CFG" 2>&1 | tee "$RUN_DIR/step2.log"

echo "===== STEP 3/3: fine-tune WITH rules (CoT2) ====="
python -u step3_finetune_with_rules.py --config "$CFG" --cot_version CoT2 2>&1 | tee "$RUN_DIR/step3.log"

echo SUCCESS > "$RUN_DIR/status.txt"
echo "All 3 steps complete."
date -u
RUN
```

**Detach and leave it running:** press **Ctrl+B**, release, then **D**. The job
keeps going after you disconnect or shut down the laptop. Do **not** press
Ctrl+C in this pane — that kills training.

To run just one step instead of all three, run its single `python -u ...` line
(with `source .venv/bin/activate` and `export HF_TOKEN=...` set first). Steps 2
and 3 auto-resume from the newest checkpoint in their output folder if one
exists — delete `outputs/guidelineKG/finetuned_LLaMA_3.2_1B_*` first if you want
a clean retrain.

---

## 6. Monitor from a second terminal

**LOCAL POWERSHELL: open a NEW window** and connect:

```powershell
ssh riftuser@<VM_IP>
```

**In that second REMOTE SSH session:**

```bash
# REMOTE SSH (second window)
cd ~/NeSyCoT-final
RUN_DIR=$(cat logs/latest_run.txt)
cat "$RUN_DIR/status.txt"        # RUNNING / FAILED / SUCCESS
tail -n 50 -F "$RUN_DIR/pipeline.log"
```

`Ctrl+C` here stops only the `tail`, not the job. Follow one step with
`tail -F "$RUN_DIR/step3.log"`. Watch the GPU with `watch -n 5 nvidia-smi`.
Re-attach to the training pane with `tmux attach -t nesycot`.

A `FAILED` status means fix the logged error and rerun Section 5. A stopped or
terminated VM cannot keep the job running.

---

## 7. REMOTE SSH: archive the results

After `status.txt` says `SUCCESS`:

```bash
# REMOTE SSH
cd ~/NeSyCoT-final
RUN_DIR=$(cat logs/latest_run.txt); cat "$RUN_DIR/status.txt"
mkdir -p final_outputs
ARCHIVE="final_outputs/nesycot_guidelineKG_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
tar -czf "$ARCHIVE" outputs/guidelineKG logs && sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
echo "$ARCHIVE" > final_outputs/latest_archive.txt
ls -lh "$ARCHIVE"
```

Archive only after training stops, so files are not changing mid-copy.

---

## 8. LOCAL POWERSHELL: download and verify

Open a NEW **local** PowerShell window:

```powershell
# LOCAL POWERSHELL (prompt: PS C:\...\NeSy-CoT>)
cd "C:\Users\tehranim.TIB.001\Desktop\Computer-20260504T112359Z-3-001\Computer\Master Thesis\MahsaThesis-DigiStrucMed\NeSy-CoT"
$vm = "riftuser@185.165.50.63"
$base = "/home/riftuser/NeSyCoT-final"
$archive = (ssh $vm "cat $base/final_outputs/latest_archive.txt").Trim()
if (-not $archive) { throw "No archive path — run Section 7 on the VM first." }
$name = Split-Path $archive -Leaf
New-Item -ItemType Directory -Force .\cloudrift-results | Out-Null
scp "${vm}:$base/$archive" .\cloudrift-results\
scp "${vm}:$base/$archive.sha256" .\cloudrift-results\
$expected = ((Get-Content ".\cloudrift-results\$name.sha256") -split '\s+')[0]
$actual   = (Get-FileHash ".\cloudrift-results\$name" -Algorithm SHA256).Hash
if ($actual -ne $expected) { throw "Checksum mismatch — download again" }
tar -xzf ".\cloudrift-results\$name" -C .\cloudrift-results
Write-Host "Verified. Results in .\cloudrift-results\outputs\guidelineKG\"
```

**What you get** (under `cloudrift-results\outputs\guidelineKG\`):

```
base_model_results.json                          # Step 1 metrics
finetune_no_rules_results.json                   # Step 2 metrics
finetune_with_rules_CoT2_results.json            # Step 3 metrics
finetuned_LLaMA_3.2_1B_baseline/                 # Step 2 LoRA adapter + checkpoints
finetuned_LLaMA_3.2_1B_with_rules_CoT2/          # Step 3 LoRA adapter + checkpoints
```

Plus all logs under `cloudrift-results\logs\`. The adapters are LoRA weights —
reload them on top of the same base model, they are not standalone. **Verify the
download before terminating the VM.**

---

## Troubleshooting

- **403 / GatedRepoError on Step 1** — Llama access not approved for your token's
  account. Do Section 0a, re-run the `huggingface-cli download ... config.json`
  check in Section 4, then rerun. The run fails fast (before training), so you
  lose almost nothing.
- **401 Unauthorized / "Invalid user token"** — the `HF_TOKEN` value is
  wrong/expired; this is a token problem, not a Llama-access problem. Check with
  `huggingface-cli whoami`. `HF_TOKEN` **overrides** `huggingface-cli login`, so
  fix the env var itself: copy a fresh **Read** token from
  <https://huggingface.co/settings/tokens> and redo the `read`/`export` in
  Section 4. Sanity-check with `echo "len=${#HF_TOKEN} starts=${HF_TOKEN:0:3}"`
  (expect ~37 chars, `hf_`). A common cause is a partial paste or an extra
  character at the hidden prompt.
- **No kernel image / `sm_70` missing** — wrong PyTorch. Install the exact CUDA
  12.6 wheel in Section 3 and rerun the GPU check.
- **Missing ensurepip / Python.h / compiler** — `sudo apt install -y
  python3.12-venv python3.12-dev build-essential`, recreate `.venv`.
- **CUDA out of memory** — check for other GPU processes (`nvidia-smi`); if on
  the 16 GB SXM2, set `gpu_max_memory_mb: 14000` and
  `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 4` in
  `guidelineKG-config.vm.json`, re-upload it, rerun.
- **`scp` cannot find a file** — run it from the `NeSy-CoT` folder in LOCAL
  POWERSHELL, not inside the SSH session.
- **`Connection reset by ... port 22` / `Connection closed` during upload** — the
  host dropped a large/slow transfer, not a code error. Use the gzip method in
  Section 1 (one ~180 MB file instead of ~900 MB of raw CSV) and the
  `-o ServerAliveInterval=15` keepalive flag. If a single big file still resets,
  send the CSVs one at a time and just re-run the one that failed — already
  uploaded files don't need resending.
- **`&&` / bash errors in PowerShell** — you pasted a REMOTE SSH block into a
  local window. Check the prompt: `riftuser@...:~$` = VM, `PS C:\...>` = local.
- **`REMOTE HOST IDENTIFICATION HAS CHANGED` / `Host key verification failed`** —
  not an attack. The VM was reprovisioned (or CloudRift recycled the IP), so its
  SSH host key differs from the one cached in `known_hosts`. Clear the old key
  and reconnect (type `yes` to accept the new one):
  ```powershell
  ssh-keygen -R <VM_IP>
  ```
  A changed host key usually means the VM restarted — check your files survived
  with `ssh riftuser@<VM_IP> "ls ~/NeSyCoT-final && ls ~/NeSyCoT-final/.venv"`;
  if the venv/files are gone (ephemeral storage), redo Sections 1 and 3.
- **`Permission denied (publickey)`** (no password prompt) — the instance
  accepts only SSH-key auth and your public key is not authorized on it (common
  right after a rebuild). Fix it on the CloudRift side: confirm the instance is
  running and note its current IP, then attach/register your key
  (`~/.ssh/cloudrift.pub`) to the instance — via the CloudRift SSH-key settings,
  or by pasting it into `~/.ssh/authorized_keys` from the instance's web
  console. Your `~/.ssh/config` already points at the `cloudrift` key, so once
  it's authorized, `ssh riftuser@<VM_IP>` works. A rebuilt instance is a clean
  slate — redo Sections 1 and 3.
- **Archive not found in Section 8** — run Section 7 on the VM first.
