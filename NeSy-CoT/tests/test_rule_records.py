"""Run with: python -m unittest discover -s tests -p test_rule_records.py"""
import ast
from pathlib import Path
import re
import tempfile
import unittest


# Exercise the real parser without importing optional training dependencies.
SOURCE = Path(__file__).resolve().parents[1] / "utils.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
namespace = {"re": re, "Path": Path}
functions = {"parse_rule_file", "_parse_instance_line", "load_all_rules"}
exec(compile(ast.Module(body=[node for node in tree.body
                             if isinstance(node, ast.FunctionDef)
                             and node.name in functions], type_ignores=[]),
             str(SOURCE), "exec"), namespace)


class RuleRecordsTest(unittest.TestCase):
    def parse(self, body):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rule_1.txt"
            path.write_text(
                "Rule 1: Example\n\nFormal Rule:\nHead: ?a p ?b\nBody: ?a q ?b\n\n"
                "Real Instances from Knowledge Graph (2 found):\n\n"
                + body + "\n\nRule Statistics:\n- PCA Confidence: 1.0\n",
                encoding="utf-8")
            return namespace["parse_rule_file"](str(path))

    def test_multiline_literal_preserves_answer_and_facts(self):
        rule = self.parse(
            "patient has label first line\nsecond line\n"
            "third line The path is classified as POSITIVE Answer: no\n"
            "other has label text [VALID] The path is classified as POSITIVE Answer: yes")
        self.assertEqual([i["answer"] for i in rule["instances"]], ["no", "yes"])
        self.assertIn("first line second line third line", rule["instances"][0]["instance_text"])
        self.assertEqual(rule["instances"][1]["validity"], "VALID")

    def test_empty_rule_is_not_an_instance(self):
        self.assertEqual(self.parse("No matching instances found in the Knowledge Graph.")["instances"], [])

    def test_truncated_instance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unfinished instance"):
            self.parse("patient has label text The path is classified as POSITIVE")

    def test_empty_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                namespace["load_all_rules"](directory)


if __name__ == "__main__":
    unittest.main()
