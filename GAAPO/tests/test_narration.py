import json
import unittest

from GAAPO.data import Example
from GAAPO.narration import spoken_text, speech_storyboard
from GAAPO.storyboard import StoryboardEvaluator, requested_scene_count


class NarrationTests(unittest.TestCase):
    def test_known_initialisms_and_word_boundaries(self):
        self.assertEqual(spoken_text('ICD, S-ICD and ECG; ATP for VT.'),
                         'I C D, S I C D and E C G; A T P for V T.')
        self.assertEqual(spoken_text('NYHA, ACE, ARNI, HEART and ICDs'),
                         'NYHA, ACE, ARNI, HEART and ICDs')
        self.assertEqual(spoken_text(spoken_text('ICD')), 'I C D')

    def test_only_narration_changes(self):
        source = {'title': 'ICD', 'scenes': [{'narration': 'Your ICD.',
                  'visual_description': 'Show ICD.', 'on_screen_text': 'ICD'}]}
        result = speech_storyboard(source)
        self.assertEqual(result['scenes'][0]['narration'], 'Your I C D.')
        self.assertEqual(result['scenes'][0]['on_screen_text'], 'ICD')
        self.assertEqual(source['scenes'][0]['narration'], 'Your ICD.')

    def test_wrong_scene_count_penalized_and_not_exported(self):
        scene = {'scene_number': 1, 'narration': 'ICD.',
                 'visual_description': 'Device', 'on_screen_text': 'ICD'}
        value = {'title': 'Device', 'scenes': [scene, dict(scene, scene_number=2)]}
        class Model:
            def generate(self, prompt, request): return json.dumps(value)
        records = []
        evaluator = StoryboardEvaluator(Model(), result_callback=records.append)
        example = Example('a', 'Create one-scene storyboard.', {})
        _, score = evaluator.assess('Prompt', example)
        self.assertEqual(score, 0.5)
        self.assertFalse(records[-1]['scene_count_compliant'])
        self.assertNotIn('speech_ready', [r['status'] for r in records])
        value['scenes'] = [scene]
        output, score = evaluator.assess('Prompt', example)
        self.assertEqual(score, 1)
        self.assertEqual(json.loads(output)['scenes'][0]['narration'], 'ICD.')
        speech = next(r for r in records if r['status'] == 'speech_ready')
        self.assertEqual(speech['storyboard']['scenes'][0]['narration'], 'I C D.')

    def test_count_is_taken_from_request_not_reference(self):
        self.assertEqual(requested_scene_count(Example('a', 'Create 3 scenes.', {})), 3)
        self.assertEqual(requested_scene_count(Example('a',
            'Create a brief patient\u2011education scene explaining sleep.', {})), 1)
        self.assertIsNone(requested_scene_count(Example('a', 'Describe ICD.', {'scenes': [1]})))
