"""
Probability calibration for the per-folder RandomForestClassifier models mailsort trains.

Why this exists
----------------
``RandomForestClassifier.predict_proba()`` returns the fraction of trees in the forest that
voted for a class. That is a genuinely useful ranking signal, but it is not automatically a
calibrated probability: a forest reporting 0.9 for "this email belongs to folder X" does not
mean that, among every email ever scored 0.9, 90% actually belong to X. Tree ensembles are well
known to produce scores that are systematically pulled toward the middle of the [0, 1] range
(Niculescu-Mizil & Caruana, 2005, "Predicting Good Probabilities With Supervised Learning") -
so treating ``recommendation_ratio`` as a probability, and therefore as an implied error-rate
guarantee, is not justified without calibration. See mailsort.ml.evaluation for how to check
whether a chosen ``recommendation_ratio`` actually behaves the way you expect on your own data.

What calibration does, and does not, fix
-------------------------------------------
:class:`sklearn.calibration.CalibratedClassifierCV` rescales a classifier's raw scores (here via
Platt scaling - see "Method choice" below) against cross-validated held-out folds of the training
data, so that, empirically, X% of messages scored X do belong to the predicted class. It does
**not** make the underlying classifier more accurate - a poorly discriminating classifier that is
well calibrated is still a poor classifier, just an honest one about it. Use
:mod:`mailsort.ml.evaluation` to check discrimination (precision/recall) as well, not calibration
in isolation.

Why calibration is conditional, not automatic
------------------------------------------------
``CalibratedClassifierCV`` fits its rescaling function on cross-validated folds of the *same*
per-folder training data. With very few examples of a folder - a brand-new folder, or one used
rarely - those folds have too few positive examples to fit a stable rescaling function, and the
calibration step itself becomes a source of noise rather than a correction. Rather than silently
producing an unreliable "calibrated" score, mailsort only calibrates a folder's model when *both*
classes (belongs to this folder / does not) have at least ``min_samples_per_class_for_calibration``
training examples (default: 20). This is a conservative, configurable, empirically-motivated
floor, not a theorem - it roughly ensures each cross-validation fold still has a handful of
positive examples to calibrate against (20 examples / 5 folds = 4 per fold at the default
settings). Folders that do not meet it keep their raw, uncalibrated classifier score instead -
see :data:`mailsort.results.ScoreType`, which every :class:`mailsort.results.Prediction` carries
explicitly, so a caller never has to guess which kind of score they are looking at.

Method choice: sigmoid (Platt scaling) over isotonic regression
---------------------------------------------------------------
scikit-learn's isotonic option is more flexible but needs more calibration data to avoid
overfitting - the scikit-learn user guide on calibration, echoing Niculescu-Mizil & Caruana
(2005), recommends it only once at least on the order of 1000 calibration samples are available,
and prefers sigmoid/Platt scaling below that. Given the small per-folder datasets this project
expects (individual IMAP folders, not a shared corpus), mailsort always calibrates with
``method="sigmoid"`` - a 2-parameter logistic fit that stays far more stable with limited data, at
the cost of being less flexible if the true miscalibration shape is not sigmoid-like.

Limitations
-----------
- Calibration is fit and evaluated on the same kind of held-out cross-validation folds as
  precision/recall in :mod:`mailsort.ml.evaluation`, but the two are not the same check: a model
  can be well calibrated (its scores are honest) while still discriminating poorly (it cannot
  reliably tell folders apart), or vice versa. Look at both.
- "Calibrated" here means calibrated *for this training snapshot*. Retraining (``mailsort train``)
  refits from scratch, including recomputing whether a folder still meets the calibration floor.
- The floor of 20 examples per class is a pragmatic default the project ships with, informed by
  the cross-validation fold size it implies, not a value derived from this project's own data
  (there is no single correct number). If in doubt, use :mod:`mailsort.ml.evaluation` on your own
  mailbox rather than trusting the default blindly.
"""

from __future__ import annotations

import numpy as np
import pandas
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

#: Conservative default floor on the number of training examples required in *each* class
#: (belongs to the folder / does not) before a folder's classifier is calibrated at all - see the
#: module docstring for the reasoning behind this number.
DEFAULT_MIN_SAMPLES_PER_CLASS_FOR_CALIBRATION = 20

#: Upper bound on the number of stratified cross-validation folds used to calibrate a folder's
#: classifier - actual fold count is also capped by how much data that folder has, see
#: calibration_cv_folds().
DEFAULT_MAX_CALIBRATION_CV_FOLDS = 5

#: A folder's classifier is a binary "belongs here / does not" model - calibration needs both
#: classes present in the training data.
_BINARY_CLASS_COUNT = 2


def should_calibrate(y: pandas.Series, min_samples_per_class: int) -> bool:
    """
    Decide whether there is enough data to calibrate a per-folder classifier trained on `y`.

    Args:
        y: binary (0.0/1.0) training target for one folder
        min_samples_per_class: minimum number of examples required in *both* classes for
            calibration to be considered safe - see the module docstring

    Returns:
        bool: True if both classes are present and each has at least `min_samples_per_class`
            examples
    """
    counts = y.value_counts()
    return (
        len(counts) >= _BINARY_CLASS_COUNT
        and int(counts.min()) >= min_samples_per_class
    )


def calibration_cv_folds(y: pandas.Series, max_folds: int) -> int:
    """
    Number of stratified cross-validation folds to calibrate `y` with - never more than the
    smaller class has examples for (so every fold sees at least one positive and one negative
    example), and never fewer than 2 (the minimum for cross-validation to mean anything).

    Args:
        y: binary (0.0/1.0) training target for one folder
        max_folds: upper bound on the number of folds

    Returns:
        int: number of folds to use
    """
    return max(2, min(max_folds, int(y.value_counts().min())))


def train_label_classifier(
    n_estimators: int,
    random_state: int,
    bootstrap: bool,
    max_features: int,
    X: pandas.DataFrame,
    y: pandas.Series,
    calibrate: bool = True,
    min_samples_per_class_for_calibration: int = DEFAULT_MIN_SAMPLES_PER_CLASS_FOR_CALIBRATION,
    max_calibration_cv_folds: int = DEFAULT_MAX_CALIBRATION_CV_FOLDS,
) -> RandomForestClassifier | CalibratedClassifierCV:
    """
    Train a classifier for one folder, calibrating its scores if - and only if - `calibrate` is
    True and there is enough data to do so safely (see should_calibrate() and the module
    docstring).

    Args:
        n_estimators: number of estimators in the underlying random forest
        random_state: random state for initialization
        bootstrap: whether bootstrap samples are used when building trees
        X: binary encoded features stored in a pandas DataFrame
        y: binary (0.0/1.0) encoded label for this folder
        max_features: maximum number of features considered per split
        calibrate: attempt calibration at all - default: True
        min_samples_per_class_for_calibration: see should_calibrate()
        max_calibration_cv_folds: see calibration_cv_folds()

    Returns:
        RandomForestClassifier | CalibratedClassifierCV: the trained classifier for this folder.
            Both expose predict_proba() with the same shape, so callers do not need to branch on
            which one they got - use is_calibrated() if the distinction matters (see
            positive_class_probability(), which already does).
    """
    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        bootstrap=bootstrap,
        max_features=max_features,
    )
    if not calibrate or not should_calibrate(y, min_samples_per_class_for_calibration):
        return forest.fit(X=X, y=y)
    calibrated = CalibratedClassifierCV(
        estimator=forest,
        method="sigmoid",
        cv=calibration_cv_folds(y, max_calibration_cv_folds),
    )
    return calibrated.fit(X=X, y=y)


def is_calibrated(model: RandomForestClassifier | CalibratedClassifierCV) -> bool:
    """Whether `model` (as returned by train_label_classifier()) is a calibrated classifier."""
    return isinstance(model, CalibratedClassifierCV)


def positive_class_probability(
    model: RandomForestClassifier | CalibratedClassifierCV, X: pandas.DataFrame
) -> np.ndarray:
    """
    Probability of the positive class ("belongs to this folder") for every row of `X` - a
    calibrated probability if `model` is a CalibratedClassifierCV, an uncalibrated classifier
    score otherwise (see is_calibrated() and the module docstring for what that distinction means
    in practice).

    Handles the edge case where a folder's training data contained only one class (e.g. every
    email in the training set happened to carry that label): predict_proba() then returns a
    single column instead of two.

    Args:
        model: a classifier trained by train_label_classifier()
        X: binary encoded features to score

    Returns:
        numpy.ndarray: one probability per row of `X`, in [0, 1]
    """
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1.0 not in classes:
        return np.zeros(len(X))
    return proba[:, classes.index(1.0)]
