import json
import unittest
from GAAPO.script_pipeline import extract_literature, run_pipeline, editing_prompt


class ScriptPipelineTests(unittest.TestCase):
    def test_original_request_sources_and_draft_reach_editor(self):
        request = 'German HFrEF superhero video for Animaker, two to three minutes.'
        sources = [extract_literature('reference.txt', b'Source evidence only.')]
        checkpoints = []
        class Editor:
            def generate(self, prompt):
                payload = json.loads(prompt.split('\n\n', 1)[1])
                self.assert_payload = payload
                return 'Scene 1: Final German script'
        editor = Editor()
        def draft(messages):
            self.assertEqual(messages[1]['role'], 'user')
            self.assertEqual(json.loads(messages[1]['content'])['original_request'], request)
            self.assertEqual(json.loads(messages[1]['content'])['literature'], sources)
            return 'Unchanged malformed HFpFH draft'
        result = run_pipeline(request, sources, draft, editor, checkpoints.append, gaapo_prompt="GAAPO winning instruction")
        self.assertEqual(editor.assert_payload['original_request'], request)
        self.assertEqual(result['gaapo_selected_prompt'], 'GAAPO winning instruction')
        self.assertEqual(editor.assert_payload['literature'], sources)
        self.assertEqual(editor.assert_payload['unreviewed_draft'], result['draft'])
        self.assertEqual(checkpoints[-1]['status'], 'completed')

    def test_editor_failure_preserves_draft(self):
        checkpoints = []
        class Editor:
            def generate(self, prompt):
                raise RuntimeError('Service unavailable')
        with self.assertRaises(RuntimeError):
            run_pipeline('Request', [{'filename': 's.txt', 'text': 'Evidence'}],
                         lambda messages: 'Original draft', Editor(), checkpoints.append, gaapo_prompt="GAAPO winning instruction")
        self.assertEqual(checkpoints[-1]['draft'], 'Original draft')
        self.assertNotIn('final_script', checkpoints[-1])

    def test_pipeline_without_literature(self):
        def draft(messages):
            self.assertIn('learned medical knowledge', messages[0]['content'])
            return 'Medical draft'
        class Editor:
            def generate(self, prompt):
                assert 'Do not claim source verification' in prompt
                assert 'without adding new medical claims' in prompt
                return 'Edited script'
        result = run_pipeline('Use only uploaded literature', [], draft, Editor(), gaapo_prompt="GAAPO winning instruction")
        self.assertEqual(result['evidence_mode'], 'fine_tuned_draft')
        self.assertEqual(result['final_script'], 'Edited script')

    def test_empty_upload_rejected(self):
        with self.assertRaises(ValueError):
            extract_literature('empty.txt', b'  ')

    def test_final_editor_receives_pronunciation_rules_and_original_draft(self):
        original = 'Narration: ARNI and SGLT2. [ON-SCREEN TEXT: ARNI, SGLT2]'
        class Editor:
            def generate(self, prompt):
                self.prompt = prompt
                return 'Narration: Arni. [ON-SCREEN TEXT: ARNI]'
        editor = Editor()
        result = run_pipeline('German script', [], lambda messages: original,
                              editor, gaapo_prompt='Use ARNI in narration.')
        self.assertIn('write ARNI as Arni', editor.prompt)
        self.assertIn('MRA as M, R, A', editor.prompt)
        self.assertIn('S, G, L, T, zwei', editor.prompt)
        self.assertIn('S, G, L, T, two', editor.prompt)
        self.assertIn('unchanged in [ON-SCREEN TEXT:', editor.prompt)
        self.assertIn('pronunciation or direct medical-generation', editor.prompt)
        self.assertEqual(result['draft'], original)
        self.assertEqual(result['gaapo_selected_prompt'], 'Use ARNI in narration.')
        self.assertIn('[ON-SCREEN TEXT: ARNI]', result['final_script'])


if __name__ == '__main__':
    unittest.main()
