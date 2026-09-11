# DigiStrucMed: Semantic-Driven Methods for Enhancing LLM Performance

Master's thesis research by **Mahsa Forghani Tehrani**, exploring how knowledge graphs, symbolic rules, and explicit semantic context can improve large language model (LLM) performance in cardiovascular patient education.

The broader application is **Semantic-Driven Techniques for Enhancing the Quality of LLM-Generated Educational Videos for Cardiovascular Diseases**. This repository contains the supporting semantic modeling, knowledge graph construction, rule mining, fine-tuning, and evaluation work. Educational video generation is the application goal; the repository is not a complete, clinically validated video-generation system.

## Motivation and research questions

The project is motivated by heart-failure education in the DigiLearn HF setting. Patient questions require understandable information grounded in clinical evidence, while questionnaire data, medical guidelines, and terminology resources differ in format and vocabulary.

The research investigates:

- How do semantic methods affect task-specific LLM performance?
- How can structured knowledge support patient-oriented information based on medical guidelines and patient context?
- How can the approach accommodate updates to medical knowledge?

The initial hypothesis is that representing the domain as a knowledge graph, learning symbolic rules, and converting those rules into Chain-of-Thought (CoT) training examples improves performance compared with fine-tuning without explicit semantic rules. Evaluation also investigates the contribution of semantic context at inference time and generalization across question formats.

## Methodology

1. **Model the domain:** design enhanced entity-relationship (EER) models and ontologies for questionnaire and guideline knowledge.
2. **Integrate the data:** preprocess questionnaire tables and guideline extraction outputs; use RDF Mapping Language (RML) mappings to construct RDF knowledge graphs.
3. **Analyze and enrich the graphs:** investigate connectivity and centrality, query with SPARQL, and connect guideline concepts to terminology resources where supported.
4. **Learn symbolic rules:** use AMIE to mine relational rules and inspect their confidence and predicate distributions.
5. **Generate training examples:** convert graph paths and rule groundings into natural-language CoT examples.
6. **Compare models:** evaluate a base model, fine-tuning without rules, and fine-tuning with rule-derived examples.
7. **Evaluate knowledge use:** use BRINK-based question answering to examine reasoning under incomplete knowledge and the effect of explicit KG context.

The guideline component includes JSON/RML graph construction and enrichment with Wikidata, including available DrugBank identifiers. Its documentation also describes a SciSpacy/UMLS prediction pipeline. These are related workflows with different input requirements; consult the component documentation before selecting a build path.

## Repository structure

| Directory | Contents and purpose |
| --- | --- |
| [`EER/`](EER/) | Questionnaire and guideline conceptual diagrams, plus the questionnaire ontology. |
| [`Questionnarie-KG/`](Questionnarie-KG/) | Questionnaire source tables, preprocessing scripts, RML mappings, RDF exports, SPARQL queries, and exploration artifacts. The directory spelling is retained from the project. |
| [`GuideLineKG/`](GuideLineKG/) | Guideline knowledge graph construction, ontologies, mappings, extracted JSON inputs, graph navigation, and Wikidata enrichment. |
| [`NeSy-CoT/`](NeSy-CoT/) | Questionnaire KG resources, AMIE rules, CoT generation, base-model evaluation, fine-tuning, and result comparison scripts. |
| [`BRINK/`](BRINK/) | BRINK benchmark code and project-specific resources for question generation, incomplete-graph construction, prediction, and evaluation. |

Component guides: [Guideline KG](GuideLineKG/kg_build/README.md), [NeSy-CoT](NeSy-CoT/README.md), and [BRINK](BRINK/README.md).

## Experimental results

The project experiments examine questionnaire graph structure, symbolic rule quality, and model performance with and without rule-based training and semantic context.

| Experiment or artifact | Result |
| --- | --- |
| Questionnaire knowledge graph | 35,252 triples |
| Extracted symbolic rules | 2,328 rules |
| Rules with PCA confidence at least 0.5 | 1,998 rules; 330 below the threshold |
| Format-matched accuracy | 42.6% without rules; 97.0% with KG rules |
| Cross-format accuracy | 53.2% |
| BRINK Hits@Any: fine-tuned model without KG context | 2.3% |
| BRINK Hits@Any: baseline with KG context | 16.3% |
| BRINK Hits@Any: fine-tuned model with semantic context | 73.3% |

The findings suggest that explicit semantic context matters alongside fine-tuning. The cross-format result highlights a generalization limitation. Format-matched accuracy and BRINK Hits@Any measure different tasks and should not be treated as interchangeable measures of clinical correctness. Consult experiment configurations, evaluation splits, sample counts, and prediction outputs when reproducing or comparing these figures.

## Getting started

This is a research repository with separate component workflows, rather than a single installable application. Run commands from the directory indicated below and use an isolated Python environment for each workflow. Python 3.12 is used in the existing run documentation. AMIE requires Java, and model fine-tuning requires a suitable GPU environment and access to the selected model.

### Guideline graph construction

From the repository root, set up the guideline environment:

```powershell
cd GuideLineKG
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r kg_build/requirements.txt
```

For the structured JSON workflow, with the expected input JSON files in place:

```powershell
python kg_build/src/run_json_batch.py kg_build/level4-dataset
```

The batch runner scopes source-specific identifiers before merging graphs. Use it for multiple documents instead of concatenating standalone RDFizer outputs. See the [guideline guide](GuideLineKG/kg_build/README.md) for input schemas, mappings, optional enrichment, and validation details. Alternative input directories also exist in this component; select the intended dataset explicitly.

### Questionnaire graph

Start with [`Questionnarie-KG/`](Questionnarie-KG/) and inspect the source tables, preprocessing scripts, and [`config.ini`](Questionnarie-KG/config.ini). The [mapping](Questionnarie-KG/MappingRules/MappingRules.ttl) defines RDF generation; the [SPARQL queries](Questionnarie-KG/KG-SPARQL-queries.sparql) and [exploration notebook](Questionnarie-KG/KG-QuestionnarieExploration.ipynb) support graph analysis. Check local data paths before running scripts.

### NeSy-CoT experiments

Follow the [NeSy-CoT guide](NeSy-CoT/README.md) for dependencies and data preparation. A local `NeSy-CoT/config.json` is required and intentionally excluded from Git. A fresh clone therefore requires a compatible configuration and regenerated or separately obtained experiment CSVs before training can run.

After configuring data paths, model access, hyperparameters, and the required training/test files, run from `NeSy-CoT/`:

```powershell
python step1_evaluate_base.py --config config.json
python step2_finetune_no_rules.py --config config.json
python step3_finetune_with_rules.py --config config.json
python compare_results.py --config config.json
```

Keep the evaluation split and question format consistent when comparing conditions. The [VM notes](NeSy-CoT/README_VM.md) document a particular experiment environment and contain machine-specific paths and settings that must be adapted.

### BRINK evaluation

Follow the [BRINK guide](BRINK/README.md) for benchmark construction, dependencies, prediction formats, and attribution. From `BRINK/`, the documented evaluator example is:

```powershell
python evaluation/evaluate_brink.py --gold evaluation/examples/example_gold.json --pred evaluation/examples/example_pred.json
```

The evaluator expects raw model output and reports answer-set metrics including Hits@Any, precision, recall, and F1. Its incomplete-knowledge setting removes direct supporting facts while retaining alternative reasoning evidence.

## Configuration and generated files

Keep credentials in ignored local configuration files or environment variables supported by the relevant script. Never embed real API keys or Hugging Face tokens in source code, notebooks, or README examples. Environment-variable support is script-specific; a `.env` file is only loaded by scripts that explicitly support it.

The root [`.gitignore`](.gitignore) excludes local secrets and selected large artifacts, including:

- `NeSy-CoT/config.json` and `.env` files.
- `NeSy-CoT/outputs/`, downloaded run bundles, and `NeSy-CoT/CoT.zip`.
- `GuideLineKG/kg_build/output/knowledge_graph.nt`.

These files may exist locally but are not supplied by a normal clone. Other generated directories may still be tracked; inspect changes before committing. Adding an ignore rule does not remove a file already present in Git history.

## Limitations and next steps

- Improve generalization across question types and formats.
- Extend and evaluate the integration of clinical-guideline knowledge with the questionnaire-based experiments.
- Assess clinical correctness and factual reliability independently of task accuracy.
- Test broader and more diverse clinical questions.
- Continue reviewing graph connectivity, terminology alignment, rule confidence, and label balance.

Guideline integration and broader generalization remain ongoing work. The repository contains guideline construction code; downstream model evaluation and clinical validation remain separate research objectives.

## Project context and acknowledgements

**Author:** Mahsa Forghani Tehrani, master's student.

**Supervision:** Prof. Dr. Maria-Esther Vidal, Prof. Dr. Sören Auer, Dr. Disha Purohit, and M.Sc. Yashrajsinh Chudasama.

The project collaborates with the medical project *Structured Digital Learning and AI Optimization for Heart Failure Education*, including Marielle Mudra, Prof. Dr. med. David Duncker, and Dr. med. Henrike Aenne Katrin Hillmann.

Supported by the DigiStrucMed Program, Else Kröner Promotionskolleg Hannover, funded by the Else Kröner-Fresenius Foundation (`2020_EKPK.20`).

## License and attribution

The repository includes an [MIT License](LICENSE), copyright 2026 Scientific Data Management Group. Preserve applicable attribution and licensing for third-party code, models, and datasets; see the [BRINK citation information](BRINK/README.md#citation) when using that component.
