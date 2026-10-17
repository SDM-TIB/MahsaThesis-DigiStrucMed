# GAAPO + GuidelineKG — university handover

A standalone question-and-answer website using the same Llama 3.2 1B Instruct
and GuidelineKG `with_rules_CoT2` adapter used by `GAAPO/app.py`.
The default mode uses the exact saved GAAPO best prompt selected by the previous
interface. Plain GuidelineKG Q&A is also available in the Mode selector.
The trained adapter is merged into the base model, exported as GGUF and quantized
to 8 bits. The webpage and inference run in one CPU process. The main application
does not need a Hugging Face token, GPU, Python environment, or remote model API.

The page uses the full thesis title from the supplied DigiStrucMed poster, a
cardiology and knowledge graph illustration, and the original DigiStrucMed,
TIB, L3S and MHH logos. All images are bundled for offline use. Logo sources and
the illustration's generation prompt are recorded in
`web/assets/ASSET-SOURCES.txt`. The illustration is conceptual, not a rendering
of live graph data.

## Start on Windows

Extract the whole folder, then **double-click `start.cmd`**. Your browser opens
at **http://localhost:7860**. Wait for “Model ready”, enter a storyboard request,
and press “Generate storyboard”. For an ordinary question, select
“Plain GuidelineKG Q&A” and press “Ask model”. Text streams into the answer panel.
Stop, copy and download controls
are included. Double-click `stop.cmd` to stop the model.

The bundled native runtime supports Windows x64. It selects a CPU backend for the
machine automatically. Docker is an alternative on Windows, not a requirement.
If the launcher reports a missing Microsoft Visual C++ runtime, run the included
official `runtime/windows/VC_redist.x64.exe` installer once. Its Microsoft digital
signature was checked when preparing this folder. The C++ runtime is already
installed on many Windows computers.

## Saved GAAPO best prompt

The page defaults to **GAAPO best prompt · storyboard**. It loads
`web/gaapo/best_prompt.txt` from the completed `vm-real-test-02` run, selected
candidate `p000027`. The original prompt is copied byte-for-byte. The applied
system message combines that prompt with the same output requirements used by
`GAAPO.models.MedicalStoryboardModel` during the original experiment.

Enter a short request, for example:

> Create one patient-education scene explaining what heart failure is.
> Highlight Heart failure on screen.

Open “GAAPO-selected prompt and provenance” to inspect the selected prompt and
its source. Downloaded responses record the selected mode, run, candidate and
prompt checksum. The bundled candidate audit, comparison and experiment files
allow verification of how the winner was selected.

This is saved-prompt inference: each request makes one local model generation.
The prompt asks for a JSON storyboard. It does not guarantee valid JSON or
correct medical content; incomplete outputs remain visible. Start with one
scene and use Extended for a larger output budget if necessary.

The previous Streamlit storyboard interface used this winner in a separate
draft-plus-external-editor workflow. This standalone version applies the saved
prompt directly to the local GuidelineKG model, following the original GAAPO
target-model message format. It does not reproduce that two-stage editing
workflow or rerun the optimizer. The original run evaluated format validity and
structure completeness; its scores do not evaluate this CPU runtime or clinical
accuracy. Plain Q&A sends only the user's question, as in the original Q&A mode.

## Start with Docker on Linux, macOS or Windows

Install Docker with the Compose plugin and start its engine. Then run from this folder:

```sh
sh start.sh
```

On Windows, choose Docker explicitly:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File start.ps1 -Docker
```

The launcher imports the bundled Linux x86_64 runtime image if needed. The model
is mounted read-only and stays available between restarts. Open http://localhost:7860.
The local application sends no questions or answers to a cloud inference service.

Stop with `sh stop.sh`, `stop.cmd`, or `docker compose down`.
Stopping does not delete the model files.

For a direct Docker Compose startup, after importing the bundled image:

```sh
docker load --input runtime/docker-images-amd64.tar
docker compose up -d --wait --pull never
```

The bundled runtime archive is for Linux x86_64. Apple Silicon and Linux arm64
can use the multi-platform image from the same pinned release; internet is needed
for that first image pull. ARM and Linux-native executables are not bundled.

## Hardware and measured speed

Recommended starting point: a modern CPU, 4 available CPU threads, 8 GB system
RAM and approximately 3 GB free disk space for the extracted handover. Docker
needs additional space for its engine and image cache. No GPU is used.

In plain Q&A on the development laptop (Intel i7-1270P, 16 GB RAM), a 128-token answer took
44.5 seconds with the original Transformers + PEFT float32 CPU runtime and about
6 seconds with the bundled native Q8 model. The first streamed text arrived in
approximately 0.15–0.4 seconds after the model was loaded. Results vary with the
CPU, Docker virtual-machine allocation, question length, other running programs,
and generated answer length. Windows native results and Docker results are
recorded separately in `reports/`.

Default question context is 4,096 tokens and there is one inference slot. Short,
standard and extended answer limits are 128, 256 and 512 output tokens. The page
marks answers stopped by a length limit; select Extended if needed. These are
upper limits, not fixed answer lengths. Every question is independent, as in the
original GuidelineKG Q&A mode. The model does not retrieve documents or browse.
GAAPO mode includes its system prompt in the same 4,096-token context, so the
available input budget and startup latency differ from plain Q&A. A separate
saved-prompt CPU measurement is included in `reports/gaapo-cpu.json`.
With the prompt cached, the two measured storyboard requests took 11.0 and
13.8 seconds with a 256-token output limit. The frontend's first GAAPO request
took 18.6 seconds, including 4.8 seconds until first text. One request reached
its output limit. These measurements verify prompt use and operation; they do
not demonstrate valid storyboard JSON or medical accuracy.

Four CPU threads performed similarly to eight on this laptop. Tune on the target
machine rather than assuming more threads always improve speed:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File start.ps1 -Threads 4 -Port 7861
```

Docker settings can be changed by copying `.env.example` to `.env`. Native Windows
settings use `start.ps1` parameters and the optional `MODEL_FILE` environment
variable; the native launcher does not read `.env`. The published Docker port
binds to localhost by default. A university server administrator can set
`BIND_ADDRESS=0.0.0.0` and place the service behind the university's HTTPS access
configuration if a network website is required.

## What is included

| File or directory | Purpose |
| --- | --- |
| `start.cmd`, `stop.cmd` | Windows start and stop buttons |
| `start.sh`, `stop.sh`, `compose.yaml` | Docker startup on other platforms |
| `models/` | Ready-to-run merged, quantized model |
| `runtime/windows/` | Official pinned Windows CPU runtime and notices |
| `runtime/docker-images-amd64.tar` | Offline Linux x86_64 Docker runtime |
| `web/` | GAAPO best-prompt and plain Q&A webpage with streaming and controls |
| `web/gaapo/` | Exact saved prompt, applied system message and selection audit |
| `adapter/` | Original trained adapter, config and model card |
| `scripts/` | Optional export, benchmark and integrity-check utilities |
| `reports/` | Export provenance, measured speed and functional checks |
| `licenses/` | Llama license and policy, llama.cpp license and attribution |
| `huggingface/` | Optional hosted Gradio/ZeroGPU deployment |
| `SHA256SUMS.json` | Integrity hashes for the delivered files |

The handover contains the saved GAAPO best-prompt inference and Q&A components.
Training data, the optimizer's external service, local credentials and the rest
of the thesis repository are not dependencies of this application.

## Optional Hugging Face website

The local application already runs without a token. For a public demonstration
link on Hugging Face, use the separate instructions in `huggingface/README.md`.
That deployment runs the original adapter through Transformers on shared
ZeroGPU hardware. A permitted `HF_TOKEN` is used only as a Space Secret to load
the gated base model. Hosting eligibility, quotas and queues still apply;
a token is not a promise of unrestricted GPU compute.

The optional Gradio interface was initialized locally and its HTTP page was
checked. A Space has not been published and actual ZeroGPU performance has not
been measured. The CPU model and Docker/native launchers are the tested delivery.

## Reproduce and verify

The main app has no Python dependencies. Python 3.10+ can optionally verify all
delivered files or benchmark the running application:

```sh
python scripts/verify_package.py
python scripts/verify_gaapo_prompt.py
python scripts/benchmark.py
python scripts/benchmark.py --mode gaapo --tokens 256 --output reports/gaapo-benchmark.json
```

The export used these existing research dependencies:
PyTorch 2.6.0, Transformers 4.57.6, PEFT 0.21.2, safetensors 0.8.0 and
llama.cpp release b11429. The original base revision is
`9213176726f574b556790deb65791e0c5aa438b6`.
Hashes and merge details are in `reports/model-export.json`.

For regeneration, install the model-export dependencies, obtain access to the
gated base model if it is not cached, and run:

```sh
python scripts/export_model.py --output merged-model
```

Using the converter from the pinned llama.cpp b11429 source release:

```sh
git clone --depth 1 --branch b11429 https://github.com/ggml-org/llama.cpp
cd llama.cpp
python convert_hf_to_gguf.py merged-model --outfile guideline-f16.gguf --outtype f16
llama-quantize guideline-f16.gguf models/Llama-GuidelineKG-1B-Q8_0.gguf Q8_0 4
```

Use absolute model/output paths if the merged-model directory is outside the
converter checkout. The Windows quantizer is also bundled at
`runtime/windows/llama-quantize.exe`. This regeneration workflow is optional;
the university can start the supplied model immediately.

The optional Dockerfile builds an image that includes the model and page:

```sh
docker build -t university-guidelineqa:1.0.0 .
docker run --rm -p 127.0.0.1:7860:8080 university-guidelineqa:1.0.0
```

The default Compose setup avoids rebuilding and avoids storing a duplicate model
inside an application image. Its release-tag form permits completely offline
imports. The inspected upstream digest is recorded in `compose.yaml` and the
Dockerfile pins that digest explicitly.

## Research limitations and attribution

This is the existing fine-tuned research model, not a newly trained or clinically
validated model. Benchmarks verified operation and speed, not clinical accuracy;
test answers include factual mistakes. Quantization, float16 export and the
llama.cpp runtime can change generated wording. The original hard
`no_repeat_ngram_size=3` constraint is not duplicated by the CPU sampler. The CPU
page uses greedy temperature-zero generation and a repetition penalty of 1.3;
the optional Transformers page retains the original hard repetition constraint.

Questions and answers are not saved by this web application. Answers stay in the
page's memory until downloaded or the page is closed. Normal runtime logs contain
startup, route and timing information. The delivery archive excludes local logs.

Built with Llama. The Llama 3.2 Community License, attribution notice and acceptable
use policy are provided in `licenses/`. The inference runtime is llama.cpp,
distributed under MIT with its bundled third-party notices. Read the included
model license when redistributing or hosting it.

Primary technical references:

- [llama.cpp CPU server and static webpages](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [llama.cpp Docker images](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md)
- [Model conversion and quantization](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [PEFT model merging](https://huggingface.co/docs/peft/en/developer_guides/checkpoint)
- [Hugging Face ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)
