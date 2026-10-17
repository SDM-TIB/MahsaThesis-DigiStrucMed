"""Run with python -m GAAPO.experiment --dry-run --output GAAPO/runs/dry."""
import argparse
import json
from pathlib import Path

from .core import GAAPOConfig, GAAPOOptimizer
from .data import Example, load_dataset, validate_splits
from .models import MedicalStoryboardModel, MockMedicalStoryboardModel, MockPromptGenerationLLM, configured_llm
from .storyboard import StoryboardEvaluator, LLMRubricMetric, format_validity, structure_completeness
from .run_logging import RunLog, LoggedLLM

INITIAL = ('You are a medical educational storyboard generator. Transform the user request '
           'into a brief patient-oriented storyboard in English. Return only a valid JSON object. '
           'Do not use Markdown fences, headings or introductory text. Use this exact schema: '
           '{"title":"Short title","scenes":[{"scene_number":1,"narration":"Brief narration",'
           '"visual_description":"Brief visual instructions","on_screen_text":"Short caption"}]}. '
           'Follow the requested scene count. Keep narration concise. Do not invent source quotations, '
           'page numbers or treatment instructions. The entire JSON object must be complete.')


def compare_test(optimizer, train, validation, test):
    if not optimizer.completed:
        raise RuntimeError('Test evaluation is allowed only after optimization')
    validate_splits(train, validation, test)
    evaluator = optimizer.evaluator
    return {'baseline': evaluator.score(optimizer.baseline.prompt_text, test),
            'gaapo': evaluator.score(optimizer.population[0].prompt_text, test),
            'test_examples': len(test)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='GAAPO/config.example.json')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke-test', action='store_true', help='One generation, five candidates, one validation/test example')
    parser.add_argument('--use-config-datasets', action='store_true',
                        help='In dry-run mode, use configured datasets with mock models')
    parser.add_argument('--output', default='GAAPO/runs/experiment')
    args = parser.parse_args()
    log = RunLog(args.output)
    progress = log.progress
    data = json.loads(Path(args.config).read_text(encoding='utf-8'))
    smoke = args.smoke_test or data.get('smoke_test', False)
    if smoke:
        data['gaapo'].update(population_size=5, generations=1, parents_per_generation=2)
        for settings in data['generation_methods'].values():
            settings.update(enabled=True, candidates_per_generation=1)
        data['generation_methods']['apo']['batch_size'] = 1
        data['evaluation'] = {'method': 'complete'}
    config = GAAPOConfig.from_dict(data)
    if args.dry_run:
        config.population_size, config.generations = 5, 2
        config.generation_methods = {n: {'enabled': True, 'candidates_per_generation': 1}
                                     for n in config.generation_methods}
        label = {'title': 'Synthetic fixture', 'scenes': []}
        train = tuple(Example(f'tr{i}', f'Synthetic training request {i}', label) for i in range(3))
        validation = tuple(Example(f'va{i}', f'Synthetic validation request {i}', label) for i in range(4))
        test = tuple(Example(f'te{i}', f'Synthetic test request {i}', label) for i in range(4))
        if args.use_config_datasets:
            train, validation, test = (load_dataset(data['datasets'][name])
                                       for name in ('train', 'validation', 'test'))
            validate_splits(train, validation, test)
        model, llm = MockMedicalStoryboardModel(), MockPromptGenerationLLM()
        metrics, weights = None, None
    else:
        from dotenv import load_dotenv
        # Explicit process environment wins over local files; never copy secrets into run logs.
        load_dotenv(Path(__file__).parent / '.env', override=False)
        load_dotenv(Path(__file__).resolve().parent.parent / '.env', override=False)
        train, validation, test = (load_dataset(data['datasets'][name]) for name in ('train', 'validation', 'test'))
        validate_splits(train, validation, test)
        llm = configured_llm(data['optimizer_llm'])
        progress('Loading fixed medical model and adapter ...')
        model = MedicalStoryboardModel.from_adapter(**data['target_model'])
        progress('Medical model loaded. Starting prompt optimization.')
        evaluation_settings = data.get('storyboard_evaluation', {})
        available_metrics = {'format_validity': format_validity, 'structure_completeness': structure_completeness}
        metric_names = evaluation_settings.get('metrics', ['format_validity'])
        metrics = {name: available_metrics[name] for name in metric_names}
        rubrics = data.get('storyboard_evaluation', {}).get('rubrics', {})
        if rubrics:
            judge = configured_llm(data['judge_llm'])
            metrics.update({name: LLMRubricMetric(judge, rubric) for name, rubric in rubrics.items()})
        weights = data.get('storyboard_evaluation', {}).get('weights')
    if args.dry_run:
        metric_names = data.get('storyboard_evaluation', {}).get('metrics', ['format_validity'])
        available_metrics = {'format_validity': format_validity, 'structure_completeness': structure_completeness}
        metrics = {name: available_metrics[name] for name in metric_names}
        weights = data.get('storyboard_evaluation', {}).get('weights')
    if smoke:
        config = GAAPOConfig.from_dict(data)
        train, validation, test = train[:3], validation[:1], test[:1]
        progress('Smoke test: one generation, all five methods; 9 medical-model calls in total.')
    if smoke and not args.dry_run and data['target_model'].get('max_new_tokens', 1024) < 512:
        progress('Short output budget active; inspect format diagnostics for truncated JSON.')
    validate_splits(train, validation, test)
    output = Path(args.output)
    (output / 'experiment.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    (output / 'data_split_ids.json').write_text(json.dumps({name: [e.id for e in split]
        for name, split in [('train', train), ('validation', validation), ('test', test)]}, indent=2), encoding='utf-8')
    def save_storyboard(record):
        log.write('storyboards.jsonl', record)
        if record['status'] == 'speech_ready':
            directory = output / 'speech_storyboards'
            directory.mkdir(exist_ok=True)
            (directory / f"call_{record['call']:03d}.json").write_text(
                json.dumps(record['storyboard'], ensure_ascii=False, indent=2), encoding='utf-8')

    evaluator = StoryboardEvaluator(model, metrics, weights, progress=progress,
                                    result_callback=save_storyboard)
    optimizer = GAAPOOptimizer(config, LoggedLLM(llm, log), evaluator, progress=progress,
                              checkpoint_directory=args.output)
    try:
        optimizer.optimize(data.get('initial_prompt', INITIAL), train, validation,
                           initial_variants=tuple(data.get('initial_variants', ())))
    finally:
        optimizer.save(args.output)
    progress('Optimization complete. Comparing baseline and optimized prompts on held-out test data ...')
    result = compare_test(optimizer, train, validation, test)
    result['dry_run'] = args.dry_run
    result['smoke_test'] = smoke
    optimizer.save(args.output)
    (output / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    # Persist the full run specification without secrets (keys are environment references).
    (output / 'experiment.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(json.dumps({'dry_run': args.dry_run, 'comparison': result,
                      'generations': optimizer.generation_logs}, indent=2))


if __name__ == '__main__':
    main()
