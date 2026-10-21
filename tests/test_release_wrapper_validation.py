"""Torch-free regression checks for published profiling/validation wrappers."""
import argparse
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GRID = load_module("paper_grid", "analysis/unified_model_benchmark/grid.py")
GINO = load_module("paper_gino", "analysis/unified_model_benchmark/gino.py")
REUSE = load_module("paper_reuse", "analysis/zero_shot_multiseed_20260915/validate_reuse.py")


def trainer_loader_parser(relative):
    """Read the actual trainers' literal batch/worker defaults without Torch."""
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8-sig"))
    defaults = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)):
            continue
        flag = node.args[0].value
        if flag in ("--batch", "--workers"):
            defaults[flag] = next(ast.literal_eval(keyword.value)
                                  for keyword in node.keywords if keyword.arg == "default")
    if set(defaults) != {"--batch", "--workers"}:
        raise AssertionError(f"Missing loader defaults in {relative}")
    parser = argparse.ArgumentParser()
    for flag, default in defaults.items():
        parser.add_argument(flag, type=int, default=default)
    return parser


class LoaderMetadataTests(unittest.TestCase):
    def test_effective_defaults_and_both_flag_spellings(self):
        configurations = [
            (GRID, "model/train_fixed_u_srcpos_to_c_strict_ablation.py"),
            (GRID, "model/train_operator_baselines_srcpos_to_c.py"),
            (GINO, "model/train_gino_extended_s.py"),
        ]
        for module, source in configurations:
            parser = trainer_loader_parser(source)
            for flags in [[], ["--batch", "2", "--workers", "3"],
                          ["--batch=6", "--workers=2"],
                          ["--batch", "2", "--batch=5", "--workers=1"]]:
                with self.subTest(source=source, flags=flags):
                    args = parser.parse_args(flags)
                    loader = SimpleNamespace(batch_size=args.batch, num_workers=args.workers)
                    self.assertEqual(module.loader_settings(loader),
                                     {"batch": args.batch, "workers": args.workers})


class ReuseValidationTests(unittest.TestCase):
    def setUp(self):
        self.reference = pd.DataFrame({
            "case_id": np.arange(145),
            "rel_l2_c": np.linspace(0.01, 0.05, 145),
            "mae_c": np.linspace(0.02, 0.06, 145),
            "high_rmse_c": np.linspace(0.03, 0.07, 145),
        })

    def test_shuffled_identical_ids_and_small_difference_pass(self):
        new = self.reference.sample(frac=1, random_state=7).copy()
        new["rel_l2_c"] += 1e-5
        result = REUSE.validate_metrics(new, self.reference)
        self.assertEqual(result["n_test"], 145)
        self.assertAlmostEqual(result["max_absolute_metric_difference"]["rel_l2_c"], 1e-5)

    def test_large_difference_fails(self):
        new = self.reference.copy()
        new.loc[0, "mae_c"] += 3e-5
        with self.assertRaisesRegex(RuntimeError, "tolerance"):
            REUSE.validate_metrics(new, self.reference)

    def test_duplicates_fail_even_if_both_frames_match(self):
        repeated = self.reference.copy()
        repeated.loc[1, "case_id"] = repeated.loc[0, "case_id"]
        with self.assertRaisesRegex(RuntimeError, "unique"):
            REUSE.validate_metrics(repeated, repeated)

    def test_incomplete_either_frame_fails(self):
        for side in ("new", "reference"):
            with self.subTest(side=side):
                new, ref = self.reference.copy(), self.reference.copy()
                if side == "new":
                    new = new.iloc[:-1]
                else:
                    ref = ref.iloc[:-1]
                with self.assertRaisesRegex(RuntimeError, "145"):
                    REUSE.validate_metrics(new, ref)

    def test_different_id_set_fails(self):
        new = self.reference.copy()
        new.loc[0, "case_id"] = 999
        with self.assertRaisesRegex(RuntimeError, "identical"):
            REUSE.validate_metrics(new, self.reference)

    def test_null_case_id_fails(self):
        new = self.reference.astype({"case_id": float})
        new.loc[0, "case_id"] = np.nan
        with self.assertRaisesRegex(RuntimeError, "non-null"):
            REUSE.validate_metrics(new, self.reference)

    def test_nonfinite_case_id_fails_even_if_both_frames_match(self):
        invalid = self.reference.astype({"case_id": float})
        invalid.loc[0, "case_id"] = np.inf
        with self.assertRaisesRegex(RuntimeError, "finite integer"):
            REUSE.validate_metrics(invalid, invalid)

    def test_equivalent_integer_id_dtypes_pass(self):
        new = self.reference.astype({"case_id": float})
        self.assertEqual(REUSE.validate_metrics(new, self.reference)["n_test"], 145)

    def test_missing_column_fails(self):
        with self.assertRaisesRegex(RuntimeError, "missing columns"):
            REUSE.validate_metrics(self.reference.drop(columns="mae_c"), self.reference)

    def test_every_metric_rejects_nonfinite_in_either_frame(self):
        for side in ("new", "reference"):
            for column in ("rel_l2_c", "mae_c", "high_rmse_c"):
                for invalid in (np.nan, np.inf, -np.inf):
                    with self.subTest(side=side, column=column, invalid=invalid):
                        new, ref = self.reference.copy(), self.reference.copy()
                        frame = new if side == "new" else ref
                        frame.loc[0, column] = invalid
                        with self.assertRaisesRegex(RuntimeError, "non-finite"):
                            REUSE.validate_metrics(new, ref)

    def test_nonnumeric_metric_fails(self):
        new = self.reference.astype({"mae_c": object})
        new.loc[0, "mae_c"] = "incomplete"
        with self.assertRaisesRegex(RuntimeError, "invalid metrics"):
            REUSE.validate_metrics(new, self.reference)


if __name__ == "__main__":
    unittest.main()
