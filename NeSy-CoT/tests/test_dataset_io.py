import ast
import contextlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from dataset_io import audit_dataset_csv, copy_dataset_csv, write_dataset_csv


source = Path(__file__).resolve().parents[1] / "prepare_data.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
scope = dict(pd=pd, re=re, os=os, shutil=shutil, json=json,
             write_dataset_csv=write_dataset_csv, copy_dataset_csv=copy_dataset_csv)
exec(compile(ast.Module(body=[node for node in tree.body
                             if isinstance(node, ast.FunctionDef) and node.name in
                             {"_get_first_relation", "filter_skewed_relations"}],
                        type_ignores=[]), str(source), "exec"), scope)


class DatasetIOTest(unittest.TestCase):
    def test_condition_entity_question_round_trip(self):
        question = ('Based on the rule "If occurs in assertion ?a, then condition '
                    'entity ?b." and the entailment above, is this '
                    'hasConditionEntity relationship supported?')
        frame = pd.DataFrame({"Prompt": ['###Input:\n' + question],
                              "input_text": [question], "output_text": ['yes'],
                              "Label": [1], "weight": [1.0]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            write_dataset_csv(frame, path)
            pd.testing.assert_frame_equal(frame, pd.read_csv(path))
            self.assertEqual(audit_dataset_csv(path, expected_frame=frame)["rows"], 1)

    def test_changed_text_does_not_replace_existing_file(self):
        frame = pd.DataFrame({"input_text": ['original, "quoted"'], "Label": [1]})
        corrupted = pd.DataFrame({"input_text": ['changed'], "Label": [1]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("existing file", encoding="utf-8")
            # Same number of rows, columns and labels: only the text is wrong.
            with patch.object(frame, 'to_csv', side_effect=corrupted.to_csv):
                with self.assertRaisesRegex(ValueError, "changed text"):
                    write_dataset_csv(frame, path)
            self.assertEqual(path.read_text(), "existing file")

    def test_commas_quotes_and_newlines_round_trip(self):
        frame = pd.DataFrame({"input_text": ['pregnancy, childbirth, "quoted"\nsecond line', 'ordinary'],
                              "Label": [1, 0]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            write_dataset_csv(frame, path)
            pd.testing.assert_frame_equal(frame, pd.read_csv(path))
            self.assertEqual(audit_dataset_csv(path)["rows"], 2)

    def test_invalid_label_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("original", encoding="utf-8")
            with self.assertRaises(ValueError):
                write_dataset_csv(pd.DataFrame({"Label": [2]}), path)
            self.assertEqual(path.read_text(), "original")

    def test_wrong_field_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text('input_text,Label\ntext,fragment,1\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fields"):
                audit_dataset_csv(path)

    def test_truncated_backup_cannot_replace_good_file(self):
        import csv
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'broken.csv'
            target = Path(directory) / 'good.csv'
            source.write_text('input_text,Label\n"unfinished', encoding='utf-8')
            target.write_text('input_text,Label\ncomplete,1\n', encoding='utf-8')
            with self.assertRaises(csv.Error):
                copy_dataset_csv(source, target)
            self.assertEqual(audit_dataset_csv(target)['yes'], 1)

    def test_relation_comes_from_question(self):
        self.assertEqual(scope['_get_first_relation'](
            'entity has name text, with commas. is this label relationship supported?'), 'label')
        self.assertEqual(scope['_get_first_relation'](
            'entity has name text. Does entity have label with target?'), 'label')

    def test_fresh_filter_never_restores_stale_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train_data_with_rules_CoT2.csv'
            backup = path.with_name(path.stem + '_unfiltered.csv')
            fresh = pd.DataFrame({'input_text': [
                'is this drop relationship supported?', 'is this drop relationship supported?',
                'is this keep relationship supported?', 'is this keep relationship supported?'],
                'Label': [1, 1, 1, 0]})
            fresh.to_csv(path, index=False)
            pd.DataFrame({'input_text': ['stale'], 'Label': [0]}).to_csv(backup, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                removed = scope['filter_skewed_relations'](
                    directory, threshold=90, fresh_inputs=True, csv_files=[path.name])
            self.assertEqual(removed, {'drop'})
            pd.testing.assert_frame_equal(pd.read_csv(backup), fresh)
            self.assertEqual(pd.read_csv(path)['Label'].tolist(), [1, 0])


if __name__ == '__main__':
    unittest.main()
