import ast
import contextlib
import io
from pathlib import Path
import re
import unittest
from unittest.mock import Mock

import pandas as pd
from finetune_data import (stratified_group_split, assert_disjoint, audit_frame,
                           tokenize_responses, audit_token_budgets)


class CharacterTokenizer:
    is_fast = True
    eos_token_id = 0

    def __call__(self, texts, **kwargs):
        texts = [texts] if isinstance(texts, str) else texts
        result = {"input_ids": [[ord(char) for char in text] for text in texts],
                  "attention_mask": [[1] * len(text) for text in texts]}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [[(i, i + 1) for i in range(len(text))] for text in texts]
        return result


def frame():
    samples = []
    for i in range(20):
        label = i % 2
        target = f"The path is classified as {'POSITIVE' if i % 3 else 'NEGATIVE'}. The answer is {'yes' if label else 'no'}."
        samples.append({"input_text": f"question {i}", "Label": label,
                        "output_text": target, "Prompt": f"question {i}\n###Response:\n{target}"})
    return pd.DataFrame(samples)


class TrainingDataTest(unittest.TestCase):
    def test_validation_keeps_every_row_and_groups_duplicate_questions(self):
        source = pd.concat([frame(), frame().iloc[:4]], ignore_index=True)
        train, validation = stratified_group_split(source, 0.2)
        self.assertEqual(len(train) + len(validation), len(source))
        self.assertEqual(set(train["Label"]), {0, 1})
        self.assertEqual(set(validation["Label"]), {0, 1})
        assert_disjoint(train=train, validation=validation)
        expected = source["input_text"].value_counts().to_dict()
        actual = pd.concat([train, validation])["input_text"].value_counts().to_dict()
        self.assertEqual(expected, actual)
        pd.testing.assert_frame_equal(train, stratified_group_split(source, 0.2)[0])

    def test_conflicting_labels_for_same_question_are_rejected(self):
        source = frame()
        source.loc[1, "input_text"] = source.loc[0, "input_text"]
        with self.assertRaisesRegex(ValueError, "contradictory"):
            stratified_group_split(source, 0.1)

    def test_overlap_and_missing_class_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_disjoint(train=frame(), test=frame().iloc[:2])
        with self.assertRaisesRegex(ValueError, "both yes"):
            audit_frame(frame().query("Label == 0"), "train")

    def test_mismatched_target_cannot_silently_change_ground_truth(self):
        source = frame()
        source.loc[0, "Label"] = 1
        with self.assertRaisesRegex(ValueError, "disagrees"):
            audit_frame(source, "train", require_targets=True)

    def test_loss_masks_prompt_and_keeps_complete_answer_and_eos(self):
        batch = frame().iloc[:2].to_dict("list")
        result = tokenize_responses(batch, CharacterTokenizer(), 300)
        for prompt, ids, labels in zip(batch["Prompt"], result["input_ids"], result["labels"]):
            boundary = prompt.index("###Response:") + len("###Response:")
            self.assertTrue(all(value == -100 for value in labels[:boundary]))
            self.assertEqual(labels[boundary:-1], ids[boundary:-1])
            self.assertEqual(labels[-1], 0)

    def test_over_budget_example_fails_instead_of_losing_context_or_label(self):
        with self.assertRaisesRegex(ValueError, "silently truncated"):
            tokenize_responses(frame().iloc[:2].to_dict("list"), CharacterTokenizer(), 10)

    def test_generation_budget_is_checked_against_entire_reference_answer(self):
        train, validation = stratified_group_split(frame(), 0.2)
        cfg = {"training": {"max_length": 300},
               "evaluation": {"max_input_tokens": 300, "max_new_tokens": 5}}
        with self.assertRaisesRegex(ValueError, "longest reference answer"):
            audit_token_budgets(CharacterTokenizer(), cfg, train, validation, frame(), frame())

    def test_preservation_mode_avoids_unused_second_test_split(self):
        source = Path(__file__).resolve().parents[1] / "utils.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        def sample(inst, *args, **kwargs):
            return {"Prompt": inst["instance_text"], "Label": inst["label"],
                    "_classification": inst["classification"],
                    "_answer": "yes" if inst["label"] else "no"}
        scope = dict(pd=pd, _build_rule_head_index=Mock(return_value={}),
                     _relevant_rules_for=Mock(return_value=[]), create_rule_context=Mock(return_value=""),
                     _build_cot_sample=sample, _print_split_report=Mock(),
                     _stratified_split=Mock(side_effect=AssertionError("Unexpected second split")),
                     _balance_classes=Mock(side_effect=AssertionError("Unexpected resampling")))
        exec(compile(ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                                      and node.name == "generate_training_data_with_rules"],
                                type_ignores=[]), str(source), "exec"), scope)
        instances = [{"instance_text": str(i), "label": i % 2,
                      "classification": "POSITIVE" if i % 3 else "NEGATIVE"} for i in range(10)]
        rule = {"rule_id": 1, "cot_format": "CoT2", "instances": instances}
        with contextlib.redirect_stdout(io.StringIO()):
            train, unused_test = scope["generate_training_data_with_rules"](
                {}, [], {}, [rule], excluded_keys={(1, "0"), (1, "1")}, preserve_all_instances=True)
        self.assertEqual(set(train["Prompt"]), {str(i) for i in range(2, 10)})
        self.assertTrue(unused_test.empty)


if __name__ == "__main__":
    unittest.main()
