"""Regression checks for repeatable local smoke training and evaluation."""
import ast
import contextlib
import io
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


SOURCE = Path(__file__).resolve().parents[1] / "utils.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))


def functions(names, scope):
    exec(compile(ast.Module(body=[node for node in TREE.body
                                 if isinstance(node, ast.FunctionDef)
                                 and node.name in names], type_ignores=[]),
                 str(SOURCE), "exec"), scope)
    return scope


HELPERS = functions({"_select_evaluation_rows", "_resolve_yes_no_answer"},
                    dict(np=np, re=re))
select_rows = HELPERS["_select_evaluation_rows"]
resolve_answer = HELPERS["_resolve_yes_no_answer"]


class EvaluationSubsetTest(unittest.TestCase):
    def test_balanced_subset_includes_both_labels_after_all_no_prefix(self):
        frame = pd.DataFrame({"Label": [0] * 12 + [1] * 8}, index=range(100, 120))
        subset = select_rows(frame, 4, "balanced")
        self.assertEqual(subset["Label"].value_counts().to_dict(), {0: 2, 1: 2})
        self.assertEqual(len(set(subset.index)), 4)
        pd.testing.assert_frame_equal(subset, select_rows(frame, 4, "balanced"))

    def test_minority_exhaustion_fills_budget_without_duplicate_rows(self):
        frame = pd.DataFrame({"Label": [0] * 8 + [1]})
        subset = select_rows(frame, 4, "balanced")
        self.assertEqual(len(subset), 4)
        self.assertEqual(subset["Label"].sum(), 1)
        self.assertTrue(subset.index.is_unique)

    def test_balanced_zero_and_full_budget(self):
        frame = pd.DataFrame({"Label": [0, 1, 0]})
        self.assertEqual(len(select_rows(frame, 0, "balanced")), 0)
        self.assertEqual(set(select_rows(frame, 100, "balanced").index), set(frame.index))
        pd.testing.assert_frame_equal(select_rows(frame, None), frame)

    def test_default_keeps_existing_head_sampling(self):
        frame = pd.DataFrame({"Label": [0, 0, 1, 1]})
        pd.testing.assert_frame_equal(select_rows(frame, 2), frame.head(2))

    def test_balanced_requires_valid_ground_truth(self):
        for frame in (pd.DataFrame({"Prompt": ["text"]}),
                      pd.DataFrame({"Label": [0, None]}),
                      pd.DataFrame({"Label": [0, 2]})):
            with self.assertRaises(ValueError):
                select_rows(frame, 4, "balanced")


class AnswerResolutionTest(unittest.TestCase):
    def test_unresolved_answers_count_as_wrong_for_both_ground_truth_labels(self):
        class Inputs(dict):
            def to(self, device):
                return self
        inputs = Inputs(input_ids=SimpleNamespace(shape=(1, 1)), attention_mask=[1])
        tokenizer = Mock(model_max_length=100, eos_token_id=0)
        tokenizer.return_value = inputs
        tokenizer.decode.return_value = "Yes or no?"
        model = Mock(config=SimpleNamespace(max_position_embeddings=100))
        model.generate.return_value = [[0, 0]]
        scope = dict(pd=pd, re=re, np=np, _check_ml_available=Mock(),
                     torch=SimpleNamespace(no_grad=contextlib.nullcontext),
                     accuracy_score=accuracy_score, f1_score=f1_score,
                     precision_score=precision_score, recall_score=recall_score)
        functions({"_select_evaluation_rows", "_resolve_yes_no_answer", "evaluate_model"}, scope)
        frame = pd.DataFrame({"Prompt": ["example", "other"], "Label": [0, 1]})
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = scope["evaluate_model"](model, tokenizer, frame, max_samples=None,
                                               max_new_tokens=5, require_explicit_answer=True)
        self.assertEqual(metrics["y_pred"], [-1, -1])
        self.assertEqual(metrics["accuracy"], 0)
        self.assertEqual(metrics["macro_f1"], 0)
        self.assertEqual(metrics["answer_coverage"], 0)

    def test_strict_mode_does_not_guess_from_bare_words(self):
        self.assertEqual(resolve_answer("Yes or no?", require_explicit=True), (-1, "unresolved"))
        self.assertEqual(resolve_answer("The answer is YES.", require_explicit=True), (True, "explicit"))
    def test_substrings_are_not_answers(self):
        for answer in ("Knowledge is not enough", "Yesterday", "", "The answer is nobody"):
            self.assertEqual(resolve_answer(answer), (False, "tiebreak_default_no"))

    def test_explicit_answer_overrides_other_words(self):
        self.assertEqual(resolve_answer("Yes was considered. The answer is NO."),
                         (False, "explicit"))

    def test_bare_answers_and_last_word_tiebreak(self):
        self.assertEqual(resolve_answer("YES."), (True, "bare_word"))
        self.assertEqual(resolve_answer("No, no."), (False, "bare_word"))
        self.assertEqual(resolve_answer("Yes or no? No."), (False, "tiebreak"))
        self.assertEqual(resolve_answer("No or yes? Yes."), (True, "tiebreak"))


class CheckpointPolicyTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = temporary.name
        parameter = Mock(requires_grad=True)
        parameter.numel.return_value = 100
        self.model = Mock(config=SimpleNamespace(use_cache=True))
        self.model.parameters.return_value = [parameter]
        self.trainer = Mock(model=self.model)
        self.scope = dict(
            os=os, _check_ml_available=Mock(),
            prepare_model_for_kbit_training=Mock(return_value=self.model),
            find_all_linear_names=Mock(return_value=["q_proj"]),
            LoraConfig=Mock(), get_peft_model=Mock(return_value=self.model),
            TrainingArguments=Mock(), Trainer=Mock(return_value=self.trainer),
            DataCollatorForLanguageModeling=Mock(),
            DataCollatorForSeq2Seq=Mock(), EarlyStoppingCallback=Mock(),
            get_last_checkpoint=Mock(return_value="checkpoint-10"),
        )
        functions({"fine_tune_model"}, self.scope)

    def train(self, training):
        with contextlib.redirect_stdout(io.StringIO()):
            self.scope["fine_tune_model"](self.model, Mock(), [], self.output,
                                          {"training": training})

    def test_smoke_starts_fresh_even_if_completed_checkpoint_exists(self):
        self.train({"resume_from_checkpoint": False, "save_strategy": "no"})
        self.scope["get_last_checkpoint"].assert_not_called()
        self.trainer.train.assert_called_once_with(resume_from_checkpoint=None)
        self.assertEqual(self.scope["TrainingArguments"].call_args.kwargs["save_strategy"], "no")
        self.model.save_pretrained.assert_called_once_with(self.output)

    def test_existing_runs_still_resume_by_default(self):
        self.train({})
        self.trainer.train.assert_called_once_with(resume_from_checkpoint="checkpoint-10")
        self.assertEqual(self.scope["TrainingArguments"].call_args.kwargs["save_strategy"], "steps")

    def test_checkpointing_explicitly_uses_non_reentrant_mode(self):
        self.train({"resume_from_checkpoint": False})
        self.scope["prepare_model_for_kbit_training"].assert_called_once_with(
            self.model, gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.gradient_checkpointing_enable.assert_not_called()

    def test_epoch_training_passes_validation_and_best_checkpoint_settings(self):
        validation = object()
        cfg = {"training": {"num_steps": -1, "num_train_epochs": 3,
                            "response_only_loss": True, "save_strategy": "epoch",
                            "eval_strategy": "epoch", "load_best_model_at_end": True,
                            "early_stopping_patience": 2}}
        with contextlib.redirect_stdout(io.StringIO()):
            self.scope["fine_tune_model"](self.model, Mock(), [], self.output, cfg,
                                          validation_dataset=validation)
        arguments = self.scope["TrainingArguments"].call_args.kwargs
        self.assertEqual(arguments["max_steps"], -1)
        self.assertEqual(arguments["num_train_epochs"], 3)
        self.assertTrue(arguments["load_best_model_at_end"])
        self.assertFalse(arguments["dataloader_drop_last"])
        self.assertEqual(arguments["metric_for_best_model"], "eval_loss")
        self.assertIs(self.scope["Trainer"].call_args.kwargs["eval_dataset"], validation)
        self.scope["DataCollatorForSeq2Seq"].assert_called_once()


if __name__ == "__main__":
    unittest.main()
