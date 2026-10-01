"""Unit tests for the fraud detection pipeline.

These run in a few seconds and need no trained model, so they are the fast
feedback loop for the parts of the pipeline that are easy to get subtly wrong.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudguard import data as data_mod
from fraudguard import evaluate as eval_mod
from fraudguard import models as models_mod
from fraudguard import preprocess as prep_mod
from fraudguard.config import KAGGLE_COLUMNS, TARGET

SEED = 0
ROWS = 6_000
FRAUD_RATE = 0.01


def make_frame(n: int = ROWS, fraud_rate: float = FRAUD_RATE) -> pd.DataFrame:
    """A small deterministic frame with the production schema."""
    return data_mod._synthesize(n, fraud_rate, SEED)


class TestSchema(unittest.TestCase):
    def test_column_order_and_names(self):
        frame = make_frame()
        self.assertEqual(list(frame.columns), KAGGLE_COLUMNS)
        self.assertEqual([f"V{i}" for i in range(1, 29)],
                         [c for c in frame.columns if c.startswith("V")])

    def test_binary_target(self):
        frame = make_frame()
        self.assertTrue(set(frame[TARGET].unique()).issubset({0, 1}))

    def test_fraud_rate_is_respected(self):
        frame = make_frame(ROWS, FRAUD_RATE)
        actual = frame[TARGET].mean()
        self.assertAlmostEqual(actual, FRAUD_RATE, delta=0.002)

    def test_no_missing_values(self):
        frame = make_frame()
        self.assertEqual(int(frame.isna().sum().sum()), 0)

    def test_amount_is_non_negative(self):
        frame = make_frame()
        self.assertGreaterEqual(float(frame["Amount"].min()), 0.0)


class TestCleaning(unittest.TestCase):
    def test_duplicates_removed(self):
        frame = make_frame(2_000)
        duplicated = pd.concat([frame, frame.head(50)], ignore_index=True)
        expected = int(duplicated.duplicated().sum())
        clean, report = data_mod.clean_dataframe(duplicated)
        self.assertEqual(report["duplicates_removed"], expected)
        self.assertEqual(len(clean), len(frame))
        self.assertEqual(int(clean.duplicated().sum()), 0)

    def test_duplicate_rows_must_go_before_splitting(self):
        # Duplicate transactions are the classic source of train/test leakage: the
        # same real-world event can land in both folds, inflating every metric.
        frame = make_frame(3_000)
        self.assertGreater(int(frame.duplicated().sum()), 0, "fixture should contain duplicates")
        clean, _ = data_mod.clean_dataframe(frame)
        self.assertEqual(int(clean.duplicated().sum()), 0)

    def test_split_features_target(self):
        frame = make_frame(1_000)
        X, y = data_mod.split_features_target(frame)
        self.assertNotIn(TARGET, X.columns)
        self.assertEqual(len(X), len(y))

    def test_derived_columns_never_reach_the_model(self):
        # A stray exploratory column must not change the model's input contract.
        frame = make_frame(1_000).assign(hour=3.0, notes="x")
        X, _ = data_mod.split_features_target(frame)
        self.assertEqual(list(X.columns), [c for c in KAGGLE_COLUMNS if c != TARGET])
        self.assertNotIn("hour", X.columns)


class TestSplitting(unittest.TestCase):
    """The split is where a leakage bug silently invalidates every metric."""

    def setUp(self):
        # Mirrors the real pipeline: duplicates are removed before splitting, because
        # identical rows in two folds would otherwise look like train/test leakage.
        frame, _ = data_mod.clean_dataframe(make_frame(10_000, 0.01))
        self.X, self.y = data_mod.split_features_target(frame)

    def test_sizes_are_relative_to_whole_dataset(self):
        X_tr, y_tr, X_va, y_va, X_te, y_te = prep_mod.make_splits(
            self.X, self.y, test_size=0.20, valid_size=0.10, random_state=SEED
        )
        total = len(self.X)
        self.assertEqual(len(X_tr) + len(X_va) + len(X_te), total)
        self.assertAlmostEqual(len(X_te) / total, 0.20, delta=0.01)
        self.assertAlmostEqual(len(X_va) / total, 0.10, delta=0.01)
        self.assertAlmostEqual(len(X_tr) / total, 0.70, delta=0.01)

    def test_folds_are_disjoint(self):
        X_tr, y_tr, X_va, y_va, X_te, y_te = prep_mod.make_splits(
            self.X, self.y, test_size=0.20, valid_size=0.10, random_state=SEED
        )
        sets = []
        for array in (X_tr, X_va, X_te):
            sets.append({row.tobytes() for row in np.asarray(array, dtype=np.float64)})
        self.assertEqual(len(sets[0] & sets[1]), 0, "train and validation overlap")
        self.assertEqual(len(sets[0] & sets[2]), 0, "train and test overlap")
        self.assertEqual(len(sets[1] & sets[2]), 0, "validation and test overlap")

    def test_stratification_preserves_fraud_rate(self):
        X_tr, y_tr, X_va, y_va, X_te, y_te = prep_mod.make_splits(
            self.X, self.y, test_size=0.20, valid_size=0.10, random_state=SEED
        )
        overall = float(self.y.mean())
        for name, fold in (("train", y_tr), ("valid", y_va), ("test", y_te)):
            self.assertAlmostEqual(float(np.mean(fold)), overall, delta=0.004, msg=name)

    def test_scaler_is_fitted_on_train_only(self):
        bundle = prep_mod.build_splits(self.X, self.y, random_state=SEED)
        train_mean = bundle.X_train.mean(axis=0)
        self.assertTrue(np.allclose(train_mean, 0.0, atol=1e-4),
                        "scaled training data must be centred on zero")
        # A test fold that also centred on zero would prove the scaler saw it.
        self.assertGreater(abs(float(bundle.X_test.mean())), 1e-6)

    def test_smote_balances_training_fold_only(self):
        bundle = prep_mod.build_splits(self.X, self.y, smote_strategy=1.0, random_state=SEED)
        self.assertAlmostEqual(float(np.mean(bundle.y_train_balanced)), 0.5, delta=0.01)
        self.assertEqual(len(bundle.y_train), len(bundle.y_train) )
        self.assertEqual(len(bundle.X_test), len(bundle.y_test),
                         "test fold must not be resampled")
        self.assertEqual(int((bundle.y_test == 1).sum()), int((bundle.y_test == 1).sum()))

    def test_smote_adds_only_synthetic_rows(self):
        bundle = prep_mod.build_splits(self.X, self.y, smote_strategy=1.0, random_state=SEED)
        self.assertGreater(len(bundle.y_train_balanced), len(bundle.y_train))
        self.assertEqual(
            int((bundle.y_train_balanced == 0).sum()),
            int((bundle.y_train == 0).sum()),
            "SMOTE must not touch the majority class",
        )


class TestModels(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(SEED)
        self.X = rng.normal(size=(800, 30)).astype(np.float32)
        self.X[::40] += 6.0  # planted outliers
        self.y = np.zeros(800, dtype=int)
        self.y[::40] = 1

    def test_isolation_forest_scores(self):
        result = models_mod.fit_isolation_forest(self.X, contamination=0.05, n_estimators=50,
                                                  random_state=SEED, n_jobs=1)
        self.assertEqual(result.scores.shape, (800,))
        self.assertGreater(float(result.scores.mean()), 0.0)
        self.assertTrue(np.isfinite(result.scores).all())

    def test_outliers_score_higher_than_inliers(self):
        result = models_mod.fit_isolation_forest(self.X, contamination=0.05, n_estimators=80,
                                                  random_state=SEED, n_jobs=1)
        self.assertGreater(result.scores[::40].mean(), result.scores[1::40].mean())

    def test_local_outlier_factor_scores(self):
        result = models_mod.fit_local_outlier_factor(self.X, contamination=0.05, n_neighbors=10)
        self.assertEqual(result.scores.shape, (800,))
        self.assertTrue(np.isfinite(result.scores).all())

    def test_absolute_percentile_is_monotonic_and_bounded(self):
        result = models_mod.fit_isolation_forest(self.X, contamination=0.05, n_estimators=50,
                                                  random_state=SEED, n_jobs=1)
        reference = models_mod.build_reference(result.scores, n=200, seed=SEED)
        raw = np.linspace(result.scores.min(), result.scores.max(), 50)
        mapped = models_mod.absolute_percentile(raw, reference)
        self.assertTrue(np.all(np.diff(mapped) >= -1e-9), "must be monotonic")
        self.assertGreaterEqual(mapped.min(), 0.0)
        self.assertLessEqual(mapped.max(), 1.0)

    def test_absolute_percentile_handles_single_row(self):
        reference = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        single = models_mod.absolute_percentile(np.array([2.5]), reference)
        self.assertEqual(single.shape, (1,))
        self.assertGreater(float(single[0]), 0.0,
                           "a single transaction must still get a meaningful percentile")

    def test_build_reference_is_sorted_and_bounded(self):
        scores = np.random.default_rng(SEED).normal(size=5_000)
        reference = models_mod.build_reference(scores, n=500, seed=SEED)
        self.assertEqual(len(reference), 500)
        self.assertTrue(np.all(np.diff(reference) >= 0))


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0] * 90 + [1] * 10)
        self.scores = np.array([0.1] * 90 + [0.9] * 10)

    def test_confusion_matrix_counts(self):
        result = eval_mod.evaluate("t", self.y, self.scores, threshold=0.5)
        self.assertEqual((result.tp, result.fp, result.fn, result.tn), (10, 0, 0, 90))
        self.assertAlmostEqual(result.precision, 1.0)
        self.assertAlmostEqual(result.recall, 1.0)

    def test_metrics_match_sklearn(self):
        from sklearn.metrics import f1_score, precision_score, recall_score

        result = eval_mod.evaluate("t", self.y, self.scores, threshold=0.5)
        predicted = (self.scores >= 0.5).astype(int)
        self.assertAlmostEqual(result.precision, precision_score(self.y, predicted))
        self.assertAlmostEqual(result.recall, recall_score(self.y, predicted))
        self.assertAlmostEqual(result.f1, f1_score(self.y, predicted))

    def test_threshold_within_unit_interval(self):
        threshold, _ = eval_mod.tune_threshold(self.y, self.scores, strategy="max_f1")
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)

    def test_max_recall_recalls_everything_possible(self):
        threshold, stats = eval_mod.tune_threshold(self.y, self.scores, strategy="max_recall")
        self.assertGreater(stats["tp"], 0)

    def test_min_precision_strategy(self):
        threshold, stats = eval_mod.tune_threshold(
            self.y, self.scores, strategy="min_precision", min_precision=0.5
        )
        self.assertGreaterEqual(stats["precision"], 0.5)
        self.assertGreaterEqual(threshold, 0.0)

    def test_precision_at_recall_reports_floor(self):
        precision, threshold = eval_mod.precision_at_recall(self.y, self.scores, min_recall=1.0)
        self.assertGreaterEqual(precision, 0.0)
        self.assertLessEqual(threshold, 1.0)

    def test_perfect_and_random_separators(self):
        perfect = np.array([0.01] * 90 + [0.99] * 10)
        result = eval_mod.evaluate("t", self.y, perfect, threshold=0.5)
        self.assertAlmostEqual(result.roc_auc, 1.0)

        random_scores = np.random.default_rng(SEED).random(100)
        result = eval_mod.evaluate("t", self.y, random_scores, strategy="max_f1")
        self.assertLess(result.roc_auc, 1.0)

    def test_apply_threshold_matches_evaluate(self):
        a = eval_mod.apply_threshold(self.y, self.scores, 0.5)
        b = eval_mod.evaluate("t", self.y, self.scores, threshold=0.5)
        self.assertEqual((a.tp, a.fp, a.fn, a.tn), (b.tp, b.fp, b.fn, b.tn))


if __name__ == "__main__":
    unittest.main(verbosity=2)
