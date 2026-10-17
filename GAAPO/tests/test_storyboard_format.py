import json
import unittest

from GAAPO.storyboard import format_validity, inspect_format, structure_completeness, StoryboardEvaluator


class FormatTests(unittest.TestCase):
    def setUp(self):
        self.storyboard = {'title': 'Example', 'scenes': [{'scene_number': 1,
            'narration': 'Brief narration.', 'visual_description': 'Simple diagram.', 'on_screen_text': 'Example'}]}

    def test_strict_json_scores_one(self):
        text = json.dumps(self.storyboard)
        self.assertEqual(format_validity(text, None), 1)
        self.assertAlmostEqual(structure_completeness(text, None), 1)
        self.assertEqual(inspect_format(text)['errors'], [])

    def test_wrapped_json_stays_strictly_invalid_but_gets_partial_credit(self):
        text = 'Here is the output:\n```json\n' + json.dumps(self.storyboard) + '\n```'
        self.assertEqual(format_validity(text, None), 0)
        self.assertAlmostEqual(structure_completeness(text, None), 1)
        self.assertTrue(inspect_format(text)['wrapped_json'])

    def test_schema_errors_reported(self):
        del self.storyboard['scenes'][0]['on_screen_text']
        text = json.dumps(self.storyboard)
        self.assertEqual(format_validity(text, None), 0)
        self.assertGreater(structure_completeness(text, None), 0)
        self.assertLess(structure_completeness(text, None), 1)
        self.assertIn('Scene 1 has missing or empty on_screen_text', inspect_format(text)['errors'])

    def test_plain_script_credit_is_capped(self):
        text = '**Scene 1**\nNarration: Hello.\nVisual description: A diagram.\nOn-screen text: Hello.'
        self.assertEqual(format_validity(text, None), 0)
        self.assertAlmostEqual(structure_completeness(text, None), 0.4)
        self.assertEqual(structure_completeness('Unrelated text', None), 0)

    def test_truncated_json_never_scores_as_valid_json(self):
        text = json.dumps(self.storyboard)[:-2]
        self.assertEqual(format_validity(text, None), 0)
        self.assertIsNotNone(inspect_format(text)['parse_error'])

    def test_bool_scene_number_is_invalid(self):
        self.storyboard['scenes'][0]['scene_number'] = True
        self.assertEqual(format_validity(json.dumps(self.storyboard), None), 0)

    def test_scoring_logs_diagnostics_and_scalar(self):
        from GAAPO.data import Example
        class Model:
            def generate(_, prompt, request): return '```json\n' + json.dumps(self.storyboard) + '\n```'
        records = []
        evaluator = StoryboardEvaluator(Model(),
            {'format_validity': format_validity, 'structure_completeness': structure_completeness},
            {'format_validity': 0.5, 'structure_completeness': 0.5}, result_callback=records.append)
        _, score = evaluator.assess('Prompt', Example('example', 'Request', self.storyboard))
        self.assertAlmostEqual(score, 0.5)
        self.assertTrue(records[-1]['format_diagnostics']['wrapped_json'])


if __name__ == '__main__':
    unittest.main()
