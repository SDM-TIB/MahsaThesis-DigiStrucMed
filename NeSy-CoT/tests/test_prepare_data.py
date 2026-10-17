"""Check optional CoT3 handling without importing training dependencies."""
import argparse
import ast
import contextlib
import io
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from dataset_io import write_dataset_csv


ROOT = Path(__file__).resolve().parents[1]


def load_functions(filename, names, scope):
    source = ROOT / filename
    tree = ast.parse(source.read_text(encoding="utf-8"))
    exec(compile(ast.Module(body=[node for node in tree.body
                                 if isinstance(node, ast.FunctionDef)
                                 and node.name in names], type_ignores=[]),
                 str(source), "exec"), scope)


class PrepareDataTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cot2 = self.root / "CoT2"
        self.cot2.mkdir()
        self.write_rule(self.cot2)
        self.cot3 = self.root / "CoT3"
        self.output = self.root / "outputs"
        self.cfg = {
            "seed": 42, "data_dir": str(self.root),
            "output_dir": str(self.output),
            "kg_files": {"train": "train.txt", "train_processed": "train.txt",
                         "relation2id": "relations.txt"},
            "data_generation": {"include_reasoning": True},
            "rules_dir_cot2": str(self.cot2),
            "rules_dir_cot3": str(self.cot3),
        }
        self.test_keys = {(1, "held-out instance")}
        frame = pd.DataFrame({"input_text": ["example"], "Label": [1]})
        self.shared = Mock(return_value=(set(), self.test_keys,
                                         frame, frame, frame, frame))
        self.generate = Mock()
        self.scope = dict(
            write_dataset_csv=write_dataset_csv,
            argparse=argparse, Path=Path, re=re, os=os,
            load_config=Mock(return_value=self.cfg), set_all_seeds=Mock(),
            load_knowledge_graph=Mock(return_value=({}, [])),
            load_relation_mapping=Mock(return_value={}),
            generate_shared_test_set=self.shared, _generate_and_save=self.generate,
        )
        load_functions("utils.py", {"parse_rule_file", "_parse_instance_line",
                                    "load_all_rules"}, self.scope)
        load_functions("prepare_data.py", {"main"}, self.scope)

    def write_rule(self, directory):
        (directory / "rule_1.txt").write_text(
            "Rule 1: Example\n\nFormal Rule:\nHead: ?a p ?b\nBody: ?a q ?b\n\n"
            "Real Instances from Knowledge Graph (1 found):\n\n"
            "a has q b The path is classified as POSITIVE Answer: yes\n\n"
            "Rule Statistics:\n- PCA Confidence: 1.0\n", encoding="utf-8")

    def run_main(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["prepare_data.py", "--config", "example.json"]), \
                contextlib.redirect_stdout(output):
            self.scope["main"]()
        return output.getvalue()

    def assert_cot2_only(self):
        self.assertEqual(self.shared.call_args.kwargs["rules_cot3"], [])
        self.assertEqual(len(self.shared.call_args.kwargs["rules_cot2"]), 1)
        self.assertEqual([call.kwargs["train_filename"]
                          for call in self.generate.call_args_list],
                         ["train_data_without_rules.csv", "train_data_with_rules_CoT2.csv"])
        self.assertEqual(self.generate.call_args.kwargs["excluded_keys"], self.test_keys)
        self.assertEqual({path.name for path in self.output.glob("*.csv")},
                         {"test_shared_baseline.csv", "test_shared_CoT2.csv",
                          "test_eval_clean.csv"})

    def test_missing_cot3_directory_uses_cot2_for_evaluation(self):
        self.assertIn("Continuing with Baseline and CoT2 only", self.run_main())
        self.assert_cot2_only()

    def test_empty_cot3_directory_uses_cot2_for_evaluation(self):
        self.cot3.mkdir()
        (self.cot3 / "summary.txt").write_text("No rules yet")
        self.assertIn("WARNING", self.run_main())
        self.assert_cot2_only()

    def test_null_cot3_config_uses_cot2_for_evaluation(self):
        self.cfg["rules_dir_cot3"] = None
        self.assertNotIn("WARNING", self.run_main())
        self.assert_cot2_only()

    def test_available_cot3_rules_are_loaded_and_generated(self):
        self.cot3.mkdir()
        self.write_rule(self.cot3)
        self.run_main()
        self.assertEqual(len(self.shared.call_args.kwargs["rules_cot3"]), 1)
        self.assertTrue((self.output / "test_shared_CoT3.csv").is_file())
        self.assertEqual(self.generate.call_args.kwargs["train_filename"],
                         "train_data_with_rules_CoT3.csv")
        self.assertEqual(self.generate.call_args.kwargs["excluded_keys"], self.test_keys)

    def test_malformed_cot3_rule_is_not_silently_skipped(self):
        self.cot3.mkdir()
        self.write_rule(self.cot3)
        path = self.cot3 / "rule_1.txt"
        path.write_text(path.read_text(encoding="utf-8").replace(" Answer: yes", ""),
                        encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_main()
        self.shared.assert_not_called()
        self.generate.assert_not_called()

    def test_missing_required_cot2_directory_still_fails(self):
        self.cfg["rules_dir_cot2"] = str(self.root / "missing")
        with self.assertRaises(FileNotFoundError):
            self.run_main()
        self.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
