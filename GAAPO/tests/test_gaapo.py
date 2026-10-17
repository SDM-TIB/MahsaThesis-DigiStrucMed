import json
import random
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from GAAPO.core import GAAPOConfig, GAAPOOptimizer, PromptCandidate, select_best_prompts
from GAAPO.data import Example, validate_splits
from GAAPO.evaluation import evaluate
from GAAPO.experiment import compare_test
from GAAPO.generation import MUTATIONS, apo, crossover, fewshot, mutate, opro
from GAAPO.models import MockMedicalStoryboardModel, MockPromptGenerationLLM, MedicalStoryboardModel
from GAAPO.storyboard import StoryboardEvaluator, format_validity


def examples(prefix, n=4):
    return tuple(Example(f'{prefix}{i}', f'{prefix} request {i}', {'title': prefix}) for i in range(n))


class Tests(unittest.TestCase):
    def setUp(self):
        self.train, self.validation, self.test = (examples(p) for p in ('train', 'validation', 'test'))
        self.llm = MockPromptGenerationLLM()
        self.model = MockMedicalStoryboardModel()
        self.evaluator = StoryboardEvaluator(self.model)

    def test_all_eight_mutations(self):
        self.assertEqual(len(MUTATIONS), 8)
        for strategy in MUTATIONS:
            with self.subTest(strategy=strategy):
                text, metadata = mutate('Original', self.llm, random.Random(42), strategy)
                self.assertIn(strategy, self.llm.calls[-1])
                self.assertIn('Original', self.llm.calls[-1])
                self.assertEqual(metadata['mutation_type'], strategy)
                self.assertTrue(text)

    def test_midpoint_crossover(self):
        self.assertEqual(crossover('abcdef', '12345678'), 'abc5678')
        self.assertEqual(crossover('12345678', 'abcdef'), '1234def')

    def test_crossover_snaps_to_whitespace_not_mid_word(self):
        # Raw character midpoints land inside words in both parents; every word
        # in the result must still be a complete word from one of the parents,
        # never a fragment glued from two half-words (e.g. "objecene_number").
        a = 'Keep this objective intact'
        b = 'Explore this universe freely'
        result = crossover(a, b)
        for word in result.split():
            self.assertTrue(word in a.split() or word in b.split(),
                            f'{word!r} is not an intact word from either parent')

    def test_crossover_falls_back_to_character_split_without_whitespace(self):
        self.assertEqual(crossover('abcdefgh', '12345678'), 'abcd5678')

    def test_fewshot_counts_and_preservation(self):
        for n in (1, 2, 3):
            text, meta = fewshot('Original', self.train, random.Random(42), n, n)
            self.assertTrue(text.startswith('Original\n\n'))
            self.assertEqual(text.count('Example input:'), n)
            self.assertEqual(len(set(meta['example_ids'])), n)
            self.assertNotIn('validation request', text)

    def test_apo_gradients_then_rewrite(self):
        _, meta = apo('Original', self.train, self.llm, self.evaluator, random.Random(42))
        self.assertEqual(len(self.llm.calls), 2)
        self.assertIn('training errors', self.llm.calls[0])
        self.assertIn('Gradients:', self.llm.calls[1])
        self.assertEqual(len(meta['error_ids']), 4)

    def test_apo_no_errors(self):
        text, meta = apo('JSON', self.train, self.llm, self.evaluator, random.Random(42))
        self.assertEqual(text, 'JSON')
        self.assertTrue(meta['no_errors'])
        self.assertEqual(self.llm.calls, [])

    def test_opro_dropout(self):
        trajectory = [PromptCandidate(str(i), f'prompt {i}', 0, [], 'initial', i / 10) for i in range(5)]
        _, kept, meta = opro(trajectory, self.llm, random.Random(42), 0.5)
        self.assertGreater(len(kept), 0)
        self.assertLess(len(kept), 5)
        self.assertEqual(meta['trajectory_ids'], [c.id for c in kept])
        self.assertIn('score', self.llm.calls[0])

    def candidates(self):
        return [PromptCandidate(str(i), 'JSON' if i % 2 else 'Original', 1, [], 'mutation') for i in range(8)]

    def test_complete(self):
        candidates = self.candidates()
        self.assertEqual(len(evaluate(candidates, self.validation, self.evaluator, {}, random.Random(42))), 8)
        self.assertEqual(len(self.model.calls), 32)
        self.assertEqual([c.evaluation_score for c in candidates], [0, 1] * 4)

    def test_halving(self):
        candidates = self.candidates()
        best = evaluate(candidates, self.validation, self.evaluator,
                        {'method': 'successive_halving', 'batch_size': 1}, random.Random(42))
        self.assertEqual(len(best), 1)
        self.assertEqual(best[0].evaluation_score, 1)
        self.assertLess(len(self.model.calls), 32)
        self.assertEqual(sum(c.evaluation_score is None for c in candidates), 7)

    def test_bandit(self):
        best = evaluate(self.candidates(), self.validation, self.evaluator,
                        {'method': 'bandit', 'budget': 16}, random.Random(42))
        self.assertEqual(len(self.model.calls), 16)
        self.assertEqual(best[0].evaluation_score, 1)

    def test_selection(self):
        candidates = self.candidates()
        for i, c in enumerate(candidates):
            c.evaluation_score = i / 10
        self.assertEqual([c.id for c in select_best_prompts(candidates, 2)], ['7', '6'])

    def run_optimizer(self, method='complete'):
        config = GAAPOConfig(population_size=5, generations=2,
            generation_methods={n: {'enabled': True, 'candidates_per_generation': 1}
                                for n in ('opro', 'apo', 'mutation', 'crossover', 'fewshot')},
            evaluation={'method': method})
        optimizer = GAAPOOptimizer(config, self.llm, self.evaluator)
        optimizer.optimize('Original', self.train, self.validation)
        return optimizer

    def test_full_loop_and_logging(self):
        optimizer = self.run_optimizer()
        self.assertEqual(len(optimizer.history), 11)
        self.assertEqual(len(optimizer.population), 5)
        self.assertEqual(optimizer.population[0].evaluation_score, 1)
        self.assertEqual(len(optimizer.generation_logs), 3)
        self.assertEqual(set(optimizer.generation_logs[1]['methods']), {'opro', 'apo', 'mutation', 'crossover', 'fewshot'})
        with tempfile.TemporaryDirectory() as directory:
            optimizer.save(directory)
            rows = [json.loads(line) for line in (Path(directory) / 'candidates.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 11)
            self.assertIn('parent_prompts', rows[-1]['metadata'])

    def test_seed_determinism(self):
        first = self.run_optimizer()
        self.setUp()
        second = self.run_optimizer()
        self.assertEqual([asdict(c) for c in first.history], [asdict(c) for c in second.history])

    def test_test_isolation_and_baseline(self):
        optimizer = self.run_optimizer()
        self.assertFalse(any('test request' in req for _, req in self.model.calls))
        calls_before = len(self.model.calls)
        result = compare_test(optimizer, self.train, self.validation, self.test)
        self.assertEqual(result['baseline'], 0)
        self.assertEqual(result['gaapo'], 1)
        self.assertEqual(len(self.model.calls) - calls_before, 2 * len(self.test))
        for start in (calls_before, calls_before + len(self.test)):
            self.assertEqual([r for _, r in self.model.calls[start:start + len(self.test)]],
                             [e.user_request for e in self.test])

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            validate_splits(self.train, self.train)
        with self.assertRaises(ValueError):
            validate_splits(self.train, [Example('different', self.train[0].user_request, {})])

    def test_test_before_completion_rejected(self):
        with self.assertRaises(RuntimeError):
            compare_test(GAAPOOptimizer(GAAPOConfig(), self.llm, self.evaluator), self.train, self.validation, self.test)

    def test_elites_preserved_with_budgeted_evaluations(self):
        for method in ('successive_halving', 'bandit'):
            self.setUp()
            optimizer = self.run_optimizer(method)
            self.assertEqual(optimizer.population[0].evaluation_score, 1)
            self.assertTrue(all(c.metadata['validation_examples'] == len(self.validation)
                                for c in optimizer.population))

    def test_invalid_fitness_rejected(self):
        class BadEvaluator:
            def score(self, prompt, data): return float('nan')
        with self.assertRaises(ValueError):
            evaluate(self.candidates(), self.validation, BadEvaluator(), {}, random.Random(42))

    def test_format_and_custom_metrics(self):
        self.assertEqual(format_validity('{}', self.train[0]), 0)
        evaluator = StoryboardEvaluator(self.model, {'a': lambda out, e: 1, 'b': lambda out, e: 0}, {'a': 3, 'b': 1})
        self.assertEqual(evaluator.score('Original', self.train), 0.75)

    def test_model_wrapper_preserves_request(self):
        from unittest.mock import patch
        with patch('BRINK.model_inference.generate_fixed', return_value='result') as generate:
            model = MedicalStoryboardModel('tokenizer', 'fixed model')
            self.assertEqual(model.generate('meta', 'original request'), 'result')
            messages = generate.call_args.args[2]
            self.assertTrue(messages[0]['content'].startswith('meta\n'))
            self.assertIn('ICD as I C D', messages[0]['content'])
            self.assertEqual(messages[1], {'role': 'user', 'content': 'original request'})

    def test_adapter_keeps_explicit_device_map(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        from BRINK.model_inference import load_fixed_model
        torch = SimpleNamespace(float32='float32')
        peft = SimpleNamespace(PeftModel=MagicMock())
        transformers = SimpleNamespace(AutoModelForCausalLM=MagicMock(), AutoTokenizer=MagicMock())
        with patch.dict(sys.modules, {'torch': torch, 'peft': peft, 'transformers': transformers}):
            for device in ('cpu', 'cuda:0'):
                with self.subTest(device=device):
                    load_fixed_model('base', 'adapter', device=device)
                    self.assertEqual(transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs['device_map'], {'': device})
                    self.assertEqual(peft.PeftModel.from_pretrained.call_args.kwargs['device_map'], {'': device})
                    self.assertFalse(peft.PeftModel.from_pretrained.call_args.kwargs['is_trainable'])
                    peft.PeftModel.from_pretrained.return_value.eval.assert_called()

    def test_raw_storyboard_saved_even_when_scoring_fails(self):
        records = []
        def broken_metric(output, example):
            raise ValueError('Metric failure')
        evaluator = StoryboardEvaluator(self.model, {'broken': broken_metric}, result_callback=records.append)
        with self.assertRaises(ValueError):
            evaluator.assess('Original', self.train[0])
        self.assertEqual([r['status'] for r in records], ['started', 'generated'])
        self.assertEqual(records[1]['output'], 'Unstructured mock storyboard')

    def test_checkpoint_saves_generated_candidate_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            optimizer = GAAPOOptimizer(GAAPOConfig(), self.llm, self.evaluator,
                                      checkpoint_directory=directory)
            optimizer._candidate('Original', 0, 'initial')
            rows = (Path(directory) / 'candidates.jsonl').read_text().splitlines()
            self.assertEqual(json.loads(rows[0])['prompt_text'], 'Original')
            self.assertIsNone(json.loads(rows[0])['evaluation_score'])


if __name__ == '__main__':
    unittest.main()
