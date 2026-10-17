# Mathematical and statistical reference for the thesis

Every formula below is transcribed directly from the implementation (`GAAPO/core.py`,
`generation.py`, `evaluation.py`, `storyboard.py`) — not reconstructed from memory — and
every statistic is computed directly from the actual saved run artifacts
(`GAAPO/runs/document-smoke`, `GAAPO/runs/vm-real-test-01`, `GAAPO/runs/vm-real-test-02`).
Where a number can't be made rigorous (mainly: sample sizes of 4–8), that's stated
explicitly rather than dressed up.

---

## 1. Dataset descriptive statistics

| Split | $n$ | Narration length (words) | Request length (words) | Scenes/reference |
|---|---:|---|---|---|
| Train | 16 | $\bar{x}=25.9,\ s=5.3,\ [17,35]$ | $\bar{x}=26.2,\ s=7.5,\ [12,38]$ | 1 (all) |
| Validation | 4 | $\bar{x}=29.8,\ s=3.9,\ [26,35]$ | $\bar{x}=30.2,\ s=5.4,\ [24,37]$ | 1 (all) |
| Test | 8 | $\bar{x}=30.1,\ s=3.3,\ [26,35]$ | $\bar{x}=30.0,\ s=13.1,\ [18,53]$ | 1 (all) |
| **Combined** | **28** | $\bar{x}=27.6,\ s=5.1$ | — | 1 (all) |

$s$ is the population standard deviation ($\sigma$, not the sample estimator $s_{n-1}$,
to match what was computed). All 28 references are exactly one scene, derived from 3
source chat documents (347 exchanges extracted, 28 curated); the split groups whole
source modules together (train: modules 1,2,4; validation: module 3; test: modules 5,6)
to test cross-topic transfer rather than i.i.d. generalization — state this explicitly if
you report the split as a generalization test, since it is **not** a random holdout.

---

## 2. GAAPO: formal definition of the implemented algorithm

### 2.1 Notation

- $P_g = \{c_1, \dots, c_k\}$: the retained population (elite archive) at generation $g$,
  $|P_g| \le N$ where $N$ = `population_size`.
- $f(c)$: the evaluation score (fitness) of candidate $c$, $f(c) \in [0,1]$.
- $G$ = `generations`, $K$ = `parents_per_generation`.
- $D_{\text{train}}, D_{\text{val}}, D_{\text{test}}$: the three fixed example sets,
  $|D_{\text{train}}|=16$, $|D_{\text{val}}|=4$, $|D_{\text{test}}|=8$ in all real runs here.

### 2.2 Initialization

$$P_0 = \{p_0, v_1, \dots, v_m\}$$

where $p_0$ is the fixed initial prompt and $v_1,\dots,v_m$ are optional
`initial_variants` (manually supplied alternative seed prompts — $m=0$ in the first
VM run, $m=2$ in the second, after a diversity fix described in §5). Every member of
$P_0$ is scored by complete evaluation (§2.5.1) before generation 1 begins; $P_0$ is
then truncated to the top $N$ by $f$.

### 2.3 Generation loop

For $g = 1, \dots, G$:

$$\text{parents}_g = \text{top-}K\big(P_{g-1}\big), \qquad
  \text{trajectory}_g = \text{top-}20\big(\textstyle\bigcup_{i<g} P_i \cup \text{new}_i\big)$$

For each enabled operator $m \in \{\text{opro, apo, mutation, crossover, fewshot}\}$,
$n_m$ = `candidates_per_generation` new candidates are produced (parent assignment is
round-robin over $\text{parents}_g$ by candidate index). Each operator's generation
rule:

**Mutation.** Sample a strategy $s$ uniformly from 8 fixed strategies (instruction
expansion, expert-persona injection, structural variation, constraint addition,
creative backstory, task decomposition, concise optimization, role assignment).
The optimizer LLM rewrites the parent under strategy $s$:
$$c' = \mathrm{LLM}_{\text{opt}}(\text{"rewrite under } s\text{"}, \ c_{\text{parent}})$$

**Crossover.** For parents $a, b$, let $\mathrm{mid}(x) = \lfloor |x|/2 \rfloor$ and let
$\mathrm{snap}(x)$ be the index of the first whitespace character in $x$ at or after
$\mathrm{mid}(x)$ (falling back to $\mathrm{mid}(x)$ if none exists). Then:
$$c' = a[\,0 : \mathrm{snap}(a)\,] \; \Vert \; b[\,\mathrm{snap}(b) : \,]$$
No semantic segmentation — this is a character-position splice, word-boundary-safe only
(fixed during this project; originally a raw character index with no safety, which
produced severed tokens — documented in §5).

**Few-shot.** Sample $j \sim \mathrm{Uniform}\{1,2,3\}$ (bounded by `min_examples`,
`max_examples`, and $|D_{\text{train}}|$), then sample $j$ distinct examples
$e_1,\dots,e_j$ without replacement from $D_{\text{train}}$ and append them verbatim:
$$c' = c_{\text{parent}} \;\Vert\; \bigVert_{i=1}^{j}\ (\text{"Example input: "} + e_i.\text{request} + \text{"Example output: "} + e_i.\text{output})$$

**APO (Automatic Prompt Optimization).** Sample a minibatch
$B \subset D_{\text{train}}$, $|B| = \min(\text{batch\_size}, |D_{\text{train}}|)$.
Let $\mathrm{Err} = \{\,e \in B : f_{\text{example}}(c_{\text{parent}}, e) < \tau\,\}$
for error threshold $\tau$ (default $0.8$). If $\mathrm{Err} = \varnothing$, return the
parent unchanged (`no_errors` flag). Otherwise:
$$\text{critique} = \mathrm{LLM}_{\text{opt}}(\text{parent}, \mathrm{Err}), \qquad
  c' = \mathrm{LLM}_{\text{opt}}(\text{"apply gradients"}, \text{parent}, \text{critique})$$

**OPRO (Optimization by PROmpting).** Let $T$ = trajectory (top 20 historically
evaluated candidates by score). Apply dropout: keep each $c \in T$ independently with
probability $1-\delta$ (default $\delta = 0.2$); if all are dropped, keep the single
best. Sort kept entries ascending by score (weakest first, as context) and request:
$$c' = \mathrm{LLM}_{\text{opt}}\big(\text{"propose a prompt expected to outperform"}, \{(c_i, f(c_i))\}_{c_i \in \text{kept}}\big)$$
In this implementation the request text also states the empirical finding that shorter
prompts scored better in-trajectory (§5) — this is a deliberate prompt-engineering bias
added mid-project, not part of the original GAAPO paper's OPRO description.

### 2.4 Selection

$$\text{select-best}(C, n) = \text{the } n \text{ elements of } C \text{ with highest } f(c),$$
$$\text{tie-break: ascending candidate ID (deterministic)}$$

$$P_g = \text{select-best}(P_{g-1} \cup \text{new}_g,\ N)$$

This is **truncation selection with elitism**, not tournament or roulette-wheel
selection — every candidate competes globally against the entire current archive, every
generation.

### 2.5 Evaluation strategies (`evaluation.method`)

All real runs in this project used **complete evaluation only** (§2.5.1); the other two
are implemented and unit-tested but have never been exercised against the real model —
state this if you describe them as "implemented," not "evaluated."

#### 2.5.1 Complete

$$f(c) = \frac{1}{|D_{\text{val}}|}\sum_{e \in D_{\text{val}}} f_{\text{example}}(c, e)$$

Every candidate is scored on every validation example, every generation. For the
reduced-but-real config used in both real GAAPO runs analyzed here
($|D_{\text{val}}|=4$), this means initial evaluation alone costs
$|P_0| \times 4$ real model calls, and each generation costs
$\big(\sum_m n_m\big) \times 4$ calls.

#### 2.5.2 Successive halving (implemented, not yet run on real data)

With elimination fraction $\rho$ (default $0.5$) and batch size
$\beta = \max(1, \lceil 0.2\,|D_{\text{val}}| \rceil)$ by default:

$$\hat{f}^{(t)}(c) = \frac{\sum_{s \le t} \beta \cdot f_{\text{example}}(c, \text{batch}_s)}{\sum_{s \le t} \beta}
\quad\text{(cumulative, example-weighted mean over disjoint shuffled batches)}$$

$$\text{active}^{(t+1)} = \text{select-best}\Big(\text{active}^{(t)},\ \max\big(\text{survivors},\ \lceil |\text{active}^{(t)}|(1-\rho)\rceil\big)\Big)$$

Repeats until `survivors` remain or validation data is exhausted.

#### 2.5.3 Bandit / UCB-E (implemented, not yet run on real data)

Fixed-budget best-arm identification. For arm (candidate) $i$ with empirical mean
$\hat\mu_i$ over $n_i$ pulls and exploration constant $a$:

$$\text{arm}_{\text{next}} = \arg\max_i \left(\hat\mu_i + \sqrt{\dfrac{a}{n_i}}\right)$$

Each pull draws one validation example uniformly **with replacement** and scores a
single candidate on it. Explicitly documented in the code as a UCB-E-*inspired*
approximation, not a faithful reproduction of the GAAPO paper's exact batch bandit —
say so if citing it.

### 2.6 Scale of the real runs

| Run | $N$ | $G$ | $K$ | Candidates/generation | $|D_{\text{val}}|$ | $|D_{\text{test}}|$ | Real model calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| `document-smoke` | 5 | 1 | 2 | 1/method (5 total) | 1 | 1 | 9 |
| `vm-real-test-01` | 8 | 3 | 3 | 2/method (10 total) | 4 | 8 | 164 |
| `vm-real-test-02` | 8 | 3 | 3 | 2/method (10 total) | 4 | 8 | 172 |

($\Delta=8$ calls between the two VM runs is exactly the 2 added `initial_variants`
$\times$ 4 validation examples.) Full paper-scale config ($N=20, G=10$, 4
candidates/method) is defined in `config.full.local.json` but was never run to
completion — estimated $\approx$1,140 real calls, 19–24h on the available GPU, not
executed in this project.

---

## 3. Structural evaluation metrics (exact formulas)

The target model's raw text output $y$ is parsed by `inspect_format(y)`, which attempts
strict JSON parse, then (diagnostically only) attempts to extract a JSON object embedded
in surrounding text via brace-scanning. Two independently logged metrics are computed
from this:

### 3.1 `format_validity` (binary)

$$\text{format\_validity}(y) = \mathbb{1}\big[\,y \text{ parses as strict JSON} \ \wedge\ \text{not markdown-wrapped} \ \wedge\ \text{schema-complete}\,\big]$$

Schema-complete requires: a non-empty string `title`; a non-empty `scenes` array; every
scene has integer `scene_number` equal to its 1-indexed position; every scene has
non-empty string `narration`, `visual_description`, `on_screen_text`.

### 3.2 `structure_completeness` (continuous, partial credit)

For parseable JSON object $v$ with scenes array of length $L$:

$$\text{sc}(v) = \min\Big(1,\ \underbrace{0.2}_{\text{valid JSON}} + \underbrace{0.1\cdot\mathbb{1}[\text{title ok}]}_{} + \underbrace{0.1\cdot\mathbb{1}[\text{scenes nonempty}]}_{} + \sum_{(k,w)} w \cdot \frac{|\{i : \text{scene}_i.k \text{ valid}\}|}{L}\Big)$$

where $(k,w) \in \{(\texttt{scene\_number}, 0.1),\ (\texttt{narration}, 0.2),\ (\texttt{visual\_description}, 0.15),\ (\texttt{on\_screen\_text}, 0.15)\}$.
(Weights sum to exactly $1.0$: $0.2+0.1+0.1+0.1+0.2+0.15+0.15=1.0$.)

For non-JSON text, capped pattern-matched partial credit instead:
$$\text{sc}(y) = 0.1 \times \big|\{\,\text{patterns matched} \in \{\text{"scene }n\text{", narration-label, visual-label, on-screen-label}\}\,\}\big|, \quad \max 0.4$$

### 3.3 Combined per-call fitness

For the weights actually used in both real GAAPO runs,
$w_{\text{fv}} = w_{\text{sc}} = 0.5$:

$$f_{\text{example}}(c, e) = \frac{w_{\text{fv}}\cdot\text{format\_validity}(y) + w_{\text{sc}}\cdot\text{sc}(y)}{w_{\text{fv}}+w_{\text{sc}}}$$

then, if the request specifies an explicit scene count $n^*(e)$ and the output's scene
count $\ne n^*(e)$:

$$f_{\text{example}}(c,e) \leftarrow 0.5 \cdot f_{\text{example}}(c,e)$$

($y = $ the model's raw generation for prompt $c$ on example $e$; $f_{\text{example}} \in [0,1]$ always.)

---

## 4. Results: descriptive statistics across all real-model runs

$n$ = number of real model calls (every initial evaluation, candidate evaluation, APO
probe, and final test-comparison call counted individually).

| Run | $n$ | Strict-valid JSON | $\hat p$ (proportion) | 95% Wilson CI for true rate | Mean fitness $\bar f$ | $s$ | $[\min,\max]$ |
|---|---:|---:|---:|---|---:|---:|---|
| `document-smoke` | 9 | 1 | 0.111 | — (too small for a meaningful CI) | 0.200 | 0.336 | $[0,1]$ |
| `vm-real-test-01` | 164 | 0 | 0.000 | $[0, 0.0229]$ | 0.0291 | 0.0367 | $[0, 0.150]$ |
| `vm-real-test-02` | 172 | 0 | 0.000 | $[0, 0.0218]$ | 0.0289 | 0.0384 | $[0, 0.200]$ |

$s$ is the sample standard deviation ($n{-}1$ denominator). The Wilson score interval is
reported (not a simple normal approximation) because the observed proportion is exactly
at the $0$ boundary, where the normal approximation is invalid. **Correct framing for
the thesis**: "the observed rate of strictly valid JSON was 0/164 and 0/172; a Wilson
95% confidence interval bounds the true rate at no more than approximately 2.2–2.3% in
both runs" — not "the model can never produce valid JSON," which is a stronger claim
than the data supports, even though it is the practically important reading.

---

## 5. Paired comparison: baseline vs. GAAPO-selected prompt on held-out test

Both real GAAPO runs evaluate the **same** initial baseline prompt and the
GAAPO-selected best prompt on the **same 8 held-out test examples** — a matched-pairs
design. Per-example fitness, extracted directly from `storyboards.jsonl`:

**`vm-real-test-01`** (no diversity fixes; OPRO scored 0.0 in every generation; raw
character-index crossover):

| Example index | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Baseline | 0.025 | 0.000 | 0.075 | 0.025 | 0.000 | 0.025 | 0.025 | 0.025 |
| Optimized | 0.000 | 0.000 | 0.000 | 0.025 | 0.000 | 0.000 | 0.075 | 0.075 |
| Diff | $-0.025$ | $0$ | $-0.075$ | $0$ | $0$ | $-0.025$ | $+0.05$ | $+0.05$ |

$\bar d = -0.0031$, $s_d = 0.0384$ (population sd). Sign test: 2 positive, 3 negative, 3
ties, exact two-sided $p = 1.0$. Wilcoxon signed-rank (zeros dropped, $n{=}5$):
$W=7.0,\ p=0.892$; (Pratt method, zeros ranked): $W=13.0,\ p=0.771$. **No evidence of a
difference** — fully consistent with the optimized prompt performing the same as
baseline, which matches what was observed (the optimized prompt actually scored
numerically lower on average: $0.025 \to 0.0219$).

**`vm-real-test-02`** (after the diversity fixes: whitespace-safe crossover,
concision-biased OPRO request, 2 added `initial_variants`):

| Example index | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Baseline | 0.025 | 0.000 | 0.075 | 0.025 | 0.000 | 0.025 | 0.025 | 0.025 |
| Optimized | 0.119 | 0.025 | 0.075 | 0.025 | 0.025 | 0.025 | 0.025 | 0.025 |
| Diff | $+0.094$ | $+0.025$ | $0$ | $0$ | $+0.025$ | $0$ | $0$ | $0$ |

$\bar d = +0.0180$, $s_d = 0.0305$. Sign test: 3 positive, 0 negative, 5 ties, exact
two-sided $p = 0.25$. Wilcoxon (zeros dropped, $n{=}3$): $W=0.0,\ p=0.103$; (Pratt):
$W=0.0,\ p=0.085$.

**Honest statistical reading for the thesis**: with $n=8$ paired observations and
several exact ties, **no test here reaches conventional significance** ($p<0.05$) in
either run, including the second, where the mean difference is positive and
numerically the largest improvement observed all project. Report this as a descriptive,
not inferential, result: *"the mean paired difference on held-out test improved from
$-0.003$ to $+0.018$ after the generation-operator fixes, but neither is statistically
distinguishable from zero at $n=8$ ($p=0.085$–$1.0$ across sign and Wilcoxon tests)."*
Do not report a $p$-value without also reporting $n$ and the test used, given how
easily $p=0.085$ on $n=8$ with 5 ties is misread as "nearly significant" when it is
better read as "underpowered."

---

## 6. What would be needed for a confirmatory (not just descriptive) claim

State this explicitly in a limitations section — it is the single most important
statistical caveat for the thesis:

1. **Multiple random seeds.** Every run here used `random_seed=42` once. A
   confirmatory claim needs the same config re-run with $\ge 5$ different seeds to
   estimate between-run variance in the optimizer itself, separate from model-output
   noise.
2. **A larger held-out test set**, or a formal power analysis. An $n=8$ paired test
   with a baseline/optimized difference this small has very low power to detect
   anything but a large effect; report the minimum detectable effect size for $n=8$ if
   you want to defend "we could not have seen a real improvement even if one existed"
   vs. "there is in fact no improvement."
3. **A fresh held-out set for any follow-up run.** The current test set has already
   been inspected (by both of us, in this session); any further tuning that uses it
   again is no longer a held-out evaluation in the strict sense.
4. **GPU floating-point non-determinism** was directly observed this project (identical
   prompts on identical examples producing different scores across runs on a laptop
   GPU, traced to float16 CUDA matmul reduction order) — a confirmatory study should
   report hardware/precision and, ideally, repeat key calls to quantify this noise
   source directly.

---

## 7. Quick-reference: every real-model run this project executed

| Run | Hardware | $|D_{\text{val}}|,|D_{\text{test}}|$ | Generations | Real calls | Baseline test $f$ | Optimized test $f$ | Strict-valid |
|---|---|---|---:|---:|---:|---:|---:|
| `document-smoke` | local CPU | 1, 1 | 1 | 9 | 0.05 | 0.05 | 1/9 |
| `smoke-test-01` (local) | local GPU (MX550) | 1, 1 | 1 | 9 | 0.025 | 0.0 | 0/9 |
| `vm-real-test-01` | V100-SXM2-16GB | 4, 8 | 3 | 164 | 0.025 | 0.0219 | 0/164 |
| `vm-real-test-02` | V100-SXM2-16GB | 4, 8 | 3 | 172 | 0.025 | 0.0430 | 0/172 |

Source files for every number above: `GAAPO/runs/<run>/comparison.json`,
`generations.json`, `storyboards.jsonl`, `candidates.jsonl`.
