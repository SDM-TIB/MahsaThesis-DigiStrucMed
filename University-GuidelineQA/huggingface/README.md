---
title: GAAPO GuidelineKG
emoji: 📚
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.15.0
python_version: "3.12"
app_file: app.py
models:
  - meta-llama/Llama-3.2-1B-Instruct
license: llama3.2
short_description: Saved GAAPO prompt and GuidelineKG Q&A
---

# GAAPO + GuidelineKG

Built with Llama. A research demonstration of the bundled GuidelineKG LoRA adapter.
This page preserves the original Transformers decoding settings. The default
university package runs locally on CPU using llama.cpp; this optional page uses
Hugging Face ZeroGPU to provide a shareable online link.
Both pages default to the saved GAAPO best prompt from `vm-real-test-02` and
offer plain Q&A. GAAPO mode applies the exact winner plus the original
target-model output requirements as a system message, requesting storyboard
JSON. Each request makes one model generation. This is saved-prompt inference,
with the original prompt and provenance visible in the interface.

## Deployment

An optional script creates a **new private Space**, uploads the page and adapter,
and sets `HF_TOKEN` as a Secret. Install `huggingface-hub==0.36.2`, then run from
the main handover folder:

```sh
python scripts/upload_space.py --repo-id YOUR_USERNAME/Llama-GuidelineKG-QA
```

The script uses `HF_TOKEN`, an existing HF login, or a hidden token prompt.
The token needs permission to create/write the Space and read the gated model.
It does not save the token locally. Append `--public` only if you want to publish
the website, its source and the original adapter. The script requests ZeroGPU,
never a paid dedicated GPU, and refuses to overwrite an existing Space.

Alternatively, deploy manually:

1. Create a Gradio Space and select ZeroGPU hardware. This is a separate deployment;
   the main Docker package does not require a Hugging Face account or subscription.
2. Upload this directory's `app.py`, `requirements.txt` and `README.md` to the Space root.
3. Upload the handover's `adapter/` and `licenses/` directories alongside them.
   Also upload `web/gaapo/` as a directory named `gaapo/` at the Space root.
4. Accept access to `meta-llama/Llama-3.2-1B-Instruct` with the hosting account.
5. Add `HF_TOKEN` as a **Secret** in Space Settings. Use a token permitted to read
   the gated base model. Do not put the token in an uploaded file.
6. Let the Space build, then share its URL with the university.

The model loads at startup and ZeroGPU allocates GPU compute during each answer.
Select Plain GuidelineKG Q&A when comparing to the original Q&A mode; this sends
no system instruction. GAAPO mode reuses the original selected storyboard prompt.
It does not rerun optimization or use the previous external editing pipeline.

ZeroGPU hosting eligibility and visitor quotas can change. Check:
https://huggingface.co/docs/hub/spaces-zerogpu
The documentation checked on 7 October 2026 allows up to two ZeroGPU Spaces for
eligible free personal accounts (verified email and account older than 30 days).
Queues and daily quotas apply. If ZeroGPU is not available to your account, use
the included CPU Docker package; a token alone does not provide unrestricted GPU hosting.

This optional GPU deployment is prepared but has not been deployed or benchmarked
on an actual ZeroGPU allocation. The CPU handover is the tested delivery.

Questions entered here are processed on Hugging Face infrastructure. The app
does not save questions or answers to files. Model outputs are not clinically validated.
