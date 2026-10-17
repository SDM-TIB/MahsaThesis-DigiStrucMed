# GAAPO for medical storyboards

## Two-stage Animaker script interface

Run `streamlit run GAAPO/app.py` and choose **Create an Animaker script (draft + editor)**.
Load the GuidelineKG adapter and enter the complete request. Literature uploads are optional,
as PDF, DOCX, UTF-8 TXT or Markdown. PDF uploads require extractable text; scanned
documents need OCR first. Install the dependencies in `GAAPO/requirements.txt`.

The fixed fine-tuned model creates a plain-text draft from the request and extracted
literature when supplied. Without uploads, it uses its learned medical knowledge;
any older instruction requiring uploaded literature is explicitly superseded.
The editor then improves presentation using the draft without adding new medical
claims or claiming independent source verification. Uncertain or missing content
is flagged separately. The configured external editing model receives that same request,
literature and unchanged draft, and produces a script with timing, narration,
visual instructions and marked on-screen text. Language and duration follow the
request. The previous JSON/English meta-prompt is not used in this interface.
The editor defaults to `openai/gpt-oss-120b` through the Hugging Face router using
`HF_TOKEN`; model, endpoint and key environment-variable name can be changed in
the interface. Upload contents are sent to that service when generation starts.

Each attempt saves sources, prompts, settings and progress in a unique
`GAAPO/runs/script-*` directory. `draft.txt` is written before the editing call;
`final_script.txt` appears after editing succeeds. `pipeline.json` retains both
stages for comparison. Downloads contain plain text suitable for manual use with
Animaker. Source/review notes are separate from narration, and should not be spoken.

The final editing prompt includes voiceover pronunciation rules: narration uses
`Arni` for ARNI and comma-separated letters for initialisms such as `M, R, A`.
SGLT2 is written `S, G, L, T, zwei` in German or `S, G, L, T, two` in English.
Standard abbreviations stay in on-screen text. Supported full names are preferred
at first mention; unknown expansions are not invented. This rule takes priority
over conflicting pronunciation instructions in a saved GAAPO prompt. Regenerate
existing scripts to apply it, and preview the selected voice because prompts
cannot guarantee a particular speech engine's pronunciation.

The editing stage now requires a **GAAPO-selected best_prompt.txt**. The interface
loads the real run winner by default, checks it against scored candidates from a
completed run, and displays its candidate ID, operator and validation score.
The exact selected prompt and its hash are recorded with each pipeline attempt.
The order is medical-adapter draft first, then editing with that selected prompt.
Fixed workflow requirements preserve plain text, requested language and the
draft's role as medical content even if the legacy selected prompt requests JSON.

The existing `vm-real-test-01` winner was selected using structural JSON scores
for direct generation by the medical model. Its application to the editing stage
is a new experimental condition, not a validated best editing prompt. A future
GAAPO run must evaluate complete draft-to-edit outputs on the intended script
rubric before claiming this prompt is optimal for the two-stage pipeline.

This applies a saved GAAPO result; it does not launch a new search. It does not
automatically verify clinical correctness or calculate a German Flesch score.
Oversized draft inputs are rejected rather than silently dropping source text;
external-service context limits may also reject large uploads. Existing GAAPO
experiment commands and saved results remain available separately.

Implementation of [GAAPO, arXiv:2504.07157v1](https://arxiv.org/html/2504.07157v1), particularly sections 4.1–4.3. The optimized object is the **meta-prompt**; the user request and fine-tuned target model stay fixed.

## Paper correspondence

The three phases are generation → validation evaluation → ranked selection. Generation includes OPRO and APO/ProTeGi generation components, random mutation, midpoint crossover and appended few-shot examples. They share one evaluation and selection process. The archive retains the highest-scoring prompts for future parent selection. Training examples support APO and few-shot; validation scores support selection; held-out testing runs only after optimization completes.

These are a methodology implementation, not a claim to reproduce the authors' exact source code or numerical results. The paper leaves some implementation parameters unspecified; explicit choices are documented below.

- **Mutation:** sample uniformly from the eight named strategies in `generation.MUTATIONS`, pass the unchanged parent to the optimizer LLM with the selected strategy, and save the mutation name.
- **Crossover:** concatenate the first character half of one parent with the second character half of another. The split point snaps forward to the next whitespace so a child never starts life broken mid-word or mid-JSON-key (a real run produced fragments like `"...objecene_number..."` under a raw character split); there is still no semantic segmentation or LLM rewrite. Parent pairing uses the seeded random generator. With only one initial parent, self-crossover bootstraps the first generation; optional `initial_variants` allow distinct parents immediately. Swapping arguments produces the inverse child.
- **Few-shot:** sample 1–3 distinct labeled training examples per candidate and append structured input/output pairs, preserving the parent's text verbatim. The available training set must meet the configured minimum.
- **APO:** run a training minibatch through the fixed target; collect outputs scoring below the configurable threshold; request textual critiques; request a revised prompt using those critiques. No APO-specific selection loop is used. If there are no errors, return the parent with `no_errors` metadata.
- **OPRO:** take the top 20 historically evaluated prompts, randomly drop entries, sort retained entries by score, and ask the optimizer for a better prompt. The request tells the optimizer that shorter, simpler prompts have scored better in the trajectory and to prefer concision -- a real run showed OPRO's unprompted proposals trending longer and scoring 0.0 every generation, consistent with longer/more rule-heavy prompts underperforming elsewhere in that run. If dropout removes everything, retain the strongest entry. Trajectory IDs and dropout are logged. No separate OPRO evaluation loop is used.

The example configuration follows the user's equal four-candidate allocations. The paper's experimental strategy weights are mutation 0.4, APO 0.2, OPRO 0.2, few-shot 0.1, crossover 0.1. To reproduce that allocation with 20 new candidates, use counts 8, 4, 4, 2, 2 respectively. Candidate counts determine new-generation size; `population_size` limits the retained elite archive. Parents are chosen by score and assigned to candidate requests round-robin.

## Evaluation and selection

`evaluator.score(prompt, dataset)` supplies one finite scalar in [0, 1]. Complete evaluation is the default and evaluates every new candidate on all validation examples. Initial prompts are fully evaluated before the first generation.

Successive halving evaluates shuffled, disjoint validation batches and ranks cumulative example-weighted means. By default, each batch covers approximately 20% of validation and half the active candidates are removed. Set `elimination_fraction: 0.4` for the fraction used in the paper's section 4.4 experiment. Set `survivors` to control finalist count. Screening stops at that count or when validation examples are exhausted.

Bandit evaluation is explicitly an **approximation** behind the same `evaluate` interface: each arm is initialized once, then the arm maximizing `mean + sqrt(exploration / pulls)` receives a uniformly sampled validation example with replacement. Configure `budget`, `exploration`, and `survivors`. This is UCB-E-inspired fixed-budget best-arm identification, not the paper's exact batch implementation. Scores must be arithmetic means of per-example rewards for these budgeted strategies.

For both budgeted methods, finalists undergo complete validation confirmation before merging with the archive. This additional cost makes archived scores comparable across generations. Pruned candidates retain screening scores and sample counts in metadata, but have no final fitness and cannot become parents. All selection uses descending final scalar scores, with deterministic ID tie-breaking. There is no tournament selection.

## Task-specific adaptations

### Imported chat documents and revised smoke evaluation

The current local datasets replace the initial invented fixtures with a curated
subset of the three user-supplied Word chat transcripts. The importer extracts
347 exchanges and retains the complete source corpus in
`data/document_import/source_exchanges.json`. It translates/adapts 28 selected
patient-education exchanges into brief English one-scene storyboards:
16 train, 4 validation and 8 test. Original starter files are preserved under
`data/starter_backup/`.

Train uses source modules 1, 2 and 4; validation uses module 3; test uses modules
5 and 6. Whole modules stay together, including revisions. This deliberately
tests transfer between topics. It is not a random iid split. The selected set
includes educational prose adapted to video scenes, not only pre-existing
storyboard scripts. Requests in the active dataset ask for short scenes;
the faithful translated original requests, English response summaries, German
sources, paragraph indices and adaptation notes are retained in provenance.
The cited medical literature was not supplied, so source citation claims are
unverified and these ChatGPT-derived references are **not clinically reviewed**.

To reproduce extraction and adaptation using the configured Hugging Face model:

```powershell
python -m GAAPO.import_chat_documents "PATH_TO_THE_PROMPTS_FOLDER" --translate
```

Translations are cached per source exchange to avoid repeated API calls.
The curated paragraph manifest is explicit in `import_chat_documents.py`.

The local configuration now supplies an explicit JSON schema and a 256-token
output budget. It combines two independently logged structural metrics:

- `format_validity` (weight 0.5): strict valid JSON with all required fields.
- `structure_completeness` (weight 0.5): partial structural credit. A complete
  storyboard JSON object wrapped in Markdown gets structure credit but remains
  strictly invalid. Plain labeled scripts get at most 0.4 on this metric, hence
  at most 0.2 combined fitness. Missing JSON fields reduce the score.

This change is an explicit task-specific fitness adaptation; GAAPO operators
are unchanged. `storyboards.jsonl` records strict errors, wrapper detection and
JSON parsing errors separately. Outputs are never silently repaired into valid
JSON. A score of 1 measures structure, not medical correctness or full task
coverage. Check the saved raw outputs, and inspect for truncation with the
shorter token budget.

`storyboard.StoryboardEvaluator` is separate from GAAPO. It generates outputs through `MedicalStoryboardModel` and combines configurable metric callbacks and nonnegative weights into a scalar fitness. All medical content -- facts, disease/procedure/side-effect detail, everything in the storyboard -- comes from this one fine-tuned adapter; GAAPO's meta-prompt search targets how well-*structured* that output is, not where the content comes from.

The default checks JSON syntax, a nonempty title, nonempty scenes, sequential integer scene numbers, and narration/visual/on-screen text fields. **It measures structural validity only.** It does not score medical correctness, safety, completeness or relevance. Supply rubric metrics or independent callbacks for those properties. `LLMRubricMetric` accepts a judge model and a rubric; clinician-approved rubrics and references must come from your study design. A model judge's score is not clinical validation.

To enable judged metrics in JSON configuration, add `judge_llm` with the same fields as `optimizer_llm`; put rubric texts under `storyboard_evaluation.rubrics` and assign weights for every metric, including `format_validity`. API keys are read from environment variables. Alternatively construct `StoryboardEvaluator(model, metrics={...}, weights={...})` in Python.

The fixed per-call output requirements (in `MedicalStoryboardModel.generate`) also tell the
model not to append follow-up questions, rule citations, evidence-level ratings, or entailment
judgments after the JSON. The adapter's GuidelineKG fine-tuning (rule-entailment QA with CoT
reasoning -- see `NeSy-CoT/`) gives it a learned tendency to drift into that phrasing regardless
of the meta-prompt; this instruction is a direct, evidence-based mitigation for exactly that
observed failure, not a change to what generates the content.

The repository previously had QA/classification datasets and evaluators, not labeled storyboards. This implementation therefore requires separate storyboard JSON datasets and does not relabel QA examples automatically. Model loading and greedy chat inference are extracted from the existing BRINK prediction script into `BRINK/model_inference.py`, reused by both that script and the new model wrapper. No training code is called; the adapter is loaded once with training disabled. Generation uses `do_sample=False` plus `repetition_penalty`/`no_repeat_ngram_size` (see `BRINK/model_inference.generate_fixed`) to stop the degenerate repeated-phrase/duplicated-JSON loops greedy decoding alone produced on this 1B model.

## Run

### Small real-model smoke test

Use `--smoke-test` or set `"smoke_test": true` in your configuration. It uses one
generation, one candidate from each of the five methods, the first three training
examples, one APO training check, one validation example and one final test example.
This makes **nine local medical-model calls** plus four to five optimizer calls.
The generation/evaluation/selection algorithm stays the same. The token limit
stays unchanged so the smoke test can attempt complete storyboard outputs.

```powershell
python -m GAAPO.experiment --config GAAPO/config.local.json --smoke-test --output GAAPO/runs/smoke
```

The local configuration is set to smoke mode; the previous full settings are saved
as `GAAPO/config.full.local.json`. An already-running process keeps its old settings;
stop it in its terminal before starting the smoke test.

Intermediate results are written immediately:

- `steps.jsonl`: progress messages.
- `storyboards.jsonl`: each call's input/prompt, raw generated output, metric scores and fitness.
- `optimizer_calls.jsonl`: optimizer requests and returned prompt/critique text.
- `candidates.jsonl`: candidate archive updated after generation and selection.
- `generation_000.json`, `generation_001.json`: baseline and selection snapshots.
- `data_split_ids.json`: the exact examples used.
- `best_prompt.txt`, `generations.json`, `comparison.json`: best prompt and final scores.

Use a new output directory for each run; append-only step logs accumulate if a
directory is reused. These checkpoints support inspecting partial work, not automatic
resume. JSON-format scores alone and a one-example test do not establish medical quality.

Synthetic starter data can be created with `python -m GAAPO.create_starter_data`.
This writes six training, four validation and four test reference storyboards at
the example configuration's paths. It refuses to overwrite existing datasets.
Source links and the unreviewed synthetic status are recorded in
`GAAPO/data/provenance.json`. These small fixtures support pipeline development;
they are not clinician-reviewed gold labels or an independent clinical benchmark.
The data directory stays Git-ignored; the generator is retained for reproducibility.

To exercise these files using mock models and your local configuration:

```powershell
python -m GAAPO.experiment --config GAAPO/config.local.json --dry-run --use-config-datasets --output GAAPO/runs/starter
```

This runs two generations with population size five and labels the saved comparison
as a dry run. It does not require an API key or download a model. A real run still
requires an actual optimizer model in place of `REPLACE_WITH_OPTIMIZER_MODEL`.

For Hugging Face inference, use optimizer `model: openai/gpt-oss-120b`,
`base_url: https://router.huggingface.co/v1`, and `api_key_env: HF_TOKEN`.
Optional `model_env: HF_MODEL` lets the environment override the configured model.
The real-run command loads `GAAPO/.env` and then the repository-root `.env`,
preserving variables already set in the shell. Copy `.env.example` to `.env`
and set the token locally. Tokens are never placed in configuration snapshots.
Credential validation happens before the target model is loaded.

From the repository root, no dependencies or API calls are needed for the smoke run:

```powershell
python -m unittest discover -s GAAPO/tests -v
python -m GAAPO.experiment --dry-run --output GAAPO/runs/dry
```

The dry run uses population size 5, two generations, all five methods, and synthetic fixtures. Example method logs:

| Generation | OPRO | APO | Mutation | Crossover | Few-shot |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |
| 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

Mock baseline score is 0.0 and optimized score is 1.0. These scores verify plumbing only, not medical quality or research improvement. Nineteen unit tests cover all eight mutation strategies, crossover, few-shot counts, APO critiques/no-error handling, OPRO dropout, all evaluators, elitism, seed determinism, split isolation, the complete mock loop, logging and model message separation.

For real experiments:

1. Install `python -m pip install -r GAAPO/requirements.txt` in the model environment.
2. Copy `GAAPO/config.example.json` and set the actual fixed base model/adapter, optimizer model, and dataset paths. Paths are relative to the working directory.
3. Prepare three nonempty JSON arrays, each with records containing `id`, `user_request`, and `output` (a labeled storyboard object). IDs and normalized request text must be unique across all splits. Check semantic overlap separately when curating the data.
4. Configure the desired fitness metrics and API key environment variable, then run:

```powershell
python -m GAAPO.experiment --config GAAPO/config.local.json --output GAAPO/runs/medical
```

The initial manual prompt and optimized prompt are tested using the same evaluator, same in-memory target model and identical held-out test examples. Testing cannot affect generation or selection. Real smoke runs against the GuidelineKG adapter are recorded under `GAAPO/runs/document-smoke*`; see `DATASET_REPORT.md` for what they found.

Run outputs are `candidates.jsonl` (every prompt, parents, generation/method, score, rank and operator metadata), `generations.json` (method best scores and elite progress), `best_prompt.txt`, configuration snapshots and `comparison.json`. Pruned ranks are null and screening scores are recorded separately. These output directories are Git-ignored. Random seeds reproduce operator sampling, pairings, dropout and mock runs; external model/service responses and hardware execution may vary independently.
### Spoken abbreviations and scene-count checks

All real model calls include fixed output requirements for concise JSON and spoken
initialisms (for example, ICD becomes I C D in narration). These requirements apply
equally to the baseline and optimized prompts. Explicit scene counts in English
requests are checked; a mismatch halves the aggregate structural fitness, so a
two-scene answer to a one-scene request cannot receive full fitness. This is a
task-specific rule, not a GAAPO paper mechanism or a medical accuracy check.

Raw outputs remain unchanged in `storyboards.jsonl`. Only strictly valid JSON
that satisfies the requested scene count is also saved in
`speech_storyboards/call_NNN.json`. These independent copies space a curated list
of medical initialisms in narration only; titles and screen captions stay unchanged.
Use these files as the text input for your video narrator. Invalid or truncated
JSON is never silently repaired or exported. The current pipeline does not create
audio; preview the chosen voice to verify pronunciation. Pronounceable acronyms
such as NYHA, ACE and ARNI are deliberately left unchanged. Extend
`GAAPO/narration.py` for additional initialisms needed by your voice engine.
