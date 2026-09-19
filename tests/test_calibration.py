from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from mailsort.ml.calibration import (
    calibration_cv_folds,
    is_calibrated,
    positive_class_probability,
    should_calibrate,
    train_label_classifier,
)


def _binary_series(positive_count, negative_count):
    return pd.Series([1.0] * positive_count + [0.0] * negative_count)


class ShouldCalibrateTest(TestCase):
    def test_true_when_both_classes_meet_the_floor(self):
        y = _binary_series(positive_count=20, negative_count=25)

        self.assertTrue(should_calibrate(y, min_samples_per_class=20))

    def test_false_when_the_minority_class_is_below_the_floor(self):
        y = _binary_series(positive_count=19, negative_count=100)

        self.assertFalse(should_calibrate(y, min_samples_per_class=20))

    def test_false_with_only_one_class_present(self):
        y = pd.Series([1.0] * 50)

        self.assertFalse(should_calibrate(y, min_samples_per_class=20))

    def test_boundary_is_inclusive(self):
        y = _binary_series(positive_count=20, negative_count=20)

        self.assertTrue(should_calibrate(y, min_samples_per_class=20))


class CalibrationCvFoldsTest(TestCase):
    def test_capped_by_max_folds_when_data_is_plentiful(self):
        y = _binary_series(positive_count=100, negative_count=100)

        self.assertEqual(calibration_cv_folds(y, max_folds=5), 5)

    def test_capped_by_the_smaller_class_when_data_is_scarce(self):
        y = _binary_series(positive_count=3, negative_count=100)

        self.assertEqual(calibration_cv_folds(y, max_folds=5), 3)

    def test_never_goes_below_two(self):
        y = _binary_series(positive_count=1, negative_count=100)

        self.assertEqual(calibration_cv_folds(y, max_folds=5), 2)


class TrainLabelClassifierTest(TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.X = pd.DataFrame(
            rng.randint(0, 2, size=(60, 5)).astype(float),
            columns=[f"f{i}" for i in range(5)],
        )
        self.y_plenty = _binary_series(positive_count=25, negative_count=35)
        self.y_scarce = _binary_series(positive_count=3, negative_count=57)

    def test_calibrate_false_returns_a_raw_forest_even_with_plenty_of_data(self):
        model = train_label_classifier(
            n_estimators=10,
            random_state=42,
            bootstrap=True,
            max_features=2,
            X=self.X,
            y=self.y_plenty,
            calibrate=False,
        )

        self.assertIsInstance(model, RandomForestClassifier)
        self.assertFalse(is_calibrated(model))

    def test_insufficient_data_falls_back_to_a_raw_forest(self):
        model = train_label_classifier(
            n_estimators=10,
            random_state=42,
            bootstrap=True,
            max_features=2,
            X=self.X,
            y=self.y_scarce,
            calibrate=True,
            min_samples_per_class_for_calibration=20,
        )

        self.assertIsInstance(model, RandomForestClassifier)
        self.assertFalse(is_calibrated(model))

    def test_enough_data_calibrates(self):
        model = train_label_classifier(
            n_estimators=10,
            random_state=42,
            bootstrap=True,
            max_features=2,
            X=self.X,
            y=self.y_plenty,
            calibrate=True,
            min_samples_per_class_for_calibration=20,
        )

        self.assertIsInstance(model, CalibratedClassifierCV)
        self.assertTrue(is_calibrated(model))

    def test_calibrated_model_still_predicts_for_every_row(self):
        model = train_label_classifier(
            n_estimators=10,
            random_state=42,
            bootstrap=True,
            max_features=2,
            X=self.X,
            y=self.y_plenty,
            calibrate=True,
            min_samples_per_class_for_calibration=20,
        )

        proba = positive_class_probability(model, self.X)

        self.assertEqual(len(proba), len(self.X))
        self.assertTrue(((proba >= 0.0) & (proba <= 1.0)).all())


class IsCalibratedTest(TestCase):
    def test_raw_forest_is_not_calibrated(self):
        model = RandomForestClassifier(n_estimators=5).fit(
            pd.DataFrame({"f": [0, 1, 0, 1]}), pd.Series([0.0, 1.0, 0.0, 1.0])
        )

        self.assertFalse(is_calibrated(model))

    def test_calibrated_classifier_cv_is_calibrated(self):
        X = pd.DataFrame({"f": [0, 1] * 15})
        y = pd.Series([0.0, 1.0] * 15)
        model = CalibratedClassifierCV(
            estimator=RandomForestClassifier(n_estimators=5), method="sigmoid", cv=3
        ).fit(X, y)

        self.assertTrue(is_calibrated(model))


class PositiveClassProbabilityTest(TestCase):
    def test_both_classes_present(self):
        X = pd.DataFrame({"f": [0, 1, 0, 1, 0, 1]})
        y = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)

        proba = positive_class_probability(model, X)

        self.assertEqual(len(proba), len(X))
        self.assertTrue(((proba >= 0.0) & (proba <= 1.0)).all())
        # Rows with f=1 were always positive in training - should score higher on average than
        # rows with f=0, which were always negative.
        self.assertGreater(proba[X["f"] == 1].mean(), proba[X["f"] == 0].mean())

    def test_single_positive_class_present(self):
        X = pd.DataFrame({"f": [0, 1, 0, 1]})
        y = pd.Series([1.0, 1.0, 1.0, 1.0])
        model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        self.assertEqual(list(model.classes_), [1.0])

        proba = positive_class_probability(model, X)

        self.assertTrue((proba == 1.0).all())

    def test_single_negative_class_present(self):
        X = pd.DataFrame({"f": [0, 1, 0, 1]})
        y = pd.Series([0.0, 0.0, 0.0, 0.0])
        model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        self.assertEqual(list(model.classes_), [0.0])

        proba = positive_class_probability(model, X)

        self.assertTrue((proba == 0.0).all())


if __name__ == "__main__":
    import unittest

    unittest.main()
