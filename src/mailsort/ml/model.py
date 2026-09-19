from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from mailsort.ml.calibration import (
    DEFAULT_MAX_CALIBRATION_CV_FOLDS,
    DEFAULT_MIN_SAMPLES_PER_CLASS_FOR_CALIBRATION,
    is_calibrated,
    positive_class_probability,
    train_label_classifier,
)


def train_random_forest(
    n_estimators: int,
    random_state: int,
    bootstrap: bool,
    max_features: int,
    X: pandas.DataFrame,
    y: pandas.Series,
) -> RandomForestClassifier:
    """
    Train a random forest classifier

    Args:
        n_estimators (int): number of estimators of the machine learning models
        max_features (int): maximum number of features of the machine learning models
        random_state (int): random state for initialization of the machine learning models
        bootstrap (boolean): bootstrap of the machine learning models
        X (pandas.DataFrame): binary encoded features stored in a pandas dataframe
        y (pandas.Series): boolean encoded label

    Return:
        RandomForestClassifier: trained model
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        bootstrap=bootstrap,
        max_features=max_features,
    ).fit(X=X, y=y)


def fit_machine_learning_models(
    df_all_features: pandas.DataFrame,
    df_all_labels: pandas.DataFrame,
    n_estimators: int = 100,
    max_features: int = 400,
    random_state: int = 42,
    bootstrap: bool = True,
    max_workers: int | None = None,
    calibrate: bool = True,
    min_samples_per_class_for_calibration: int = DEFAULT_MIN_SAMPLES_PER_CLASS_FOR_CALIBRATION,
    max_calibration_cv_folds: int = DEFAULT_MAX_CALIBRATION_CV_FOLDS,
) -> dict[str, RandomForestClassifier]:
    """
    Train machine learning models, one per label/folder.

    Args:
        df_all_features (pandas.DataFrame): binary encoded features stored in a pandas dataframe
        df_all_labels (pandas.DataFrame): binary encoded labels stored in a pandas dataframe
        n_estimators (int): number of estimators of the machine learning models
        max_features (int): maximum number of features of the machine learning models
        random_state (int): random state for initialization of the machine learning models
        bootstrap (boolean): bootstrap of the machine learning models
        max_workers (int): maximum number of workers for the machine learning models
        calibrate (bool): calibrate each folder's classifier when there is enough data to do so
            safely - default: True. See mailsort.ml.calibration for what this means and why it is
            conditional rather than automatic.
        min_samples_per_class_for_calibration (int): see mailsort.ml.calibration.should_calibrate()
        max_calibration_cv_folds (int): see mailsort.ml.calibration.calibration_cv_folds()

    Returns:
        dict: dictionary with machine learning models with labels as keys - each value is either
            a raw RandomForestClassifier or, if calibrated, a CalibratedClassifierCV wrapping one;
            see mailsort.ml.calibration.is_calibrated()
    """
    df_training = df_all_features.drop(["email_id"], axis=1)
    if max_workers == 1:
        return {
            to_learn.split("labels_")[-1]: train_label_classifier(
                n_estimators=n_estimators,
                random_state=random_state,
                bootstrap=bootstrap,
                max_features=max_features,
                X=df_training,
                y=df_all_labels[to_learn],
                calibrate=calibrate,
                min_samples_per_class_for_calibration=min_samples_per_class_for_calibration,
                max_calibration_cv_folds=max_calibration_cv_folds,
            )
            for to_learn in tqdm(
                iterable=df_all_labels.columns.tolist(),
                desc="Train machinelearning models",
            )
        }
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures_dict = {
                to_learn.split("labels_")[-1]: exe.submit(
                    train_label_classifier,
                    n_estimators=n_estimators,
                    random_state=random_state,
                    bootstrap=bootstrap,
                    max_features=max_features,
                    X=df_training,
                    y=df_all_labels[to_learn],
                    calibrate=calibrate,
                    min_samples_per_class_for_calibration=min_samples_per_class_for_calibration,
                    max_calibration_cv_folds=max_calibration_cv_folds,
                )
                for to_learn in df_all_labels.columns.tolist()
            }
            return {
                k: v.result()
                for k, v in tqdm(
                    iterable=futures_dict.items(), desc="Train machinelearning models"
                )
            }


def _predict_top_label_per_message(
    df_features: pandas.DataFrame,
    model_dict: dict[str, RandomForestClassifier],
) -> tuple[list[str], list[str], np.ndarray, list[bool]]:
    """
    Run every per-label model in model_dict against every message in df_features and, for each
    message, determine which label's model scores it highest.

    This is the shared core behind get_predictions_from_machine_learning_models (used by the
    automatic sorting API) and score_messages_with_machine_learning_models (used by the dry-run
    recommendation API), so both agree on the exact same top label and score for a given message.

    Scores are the positive-class probability from each label's classifier (see
    mailsort.ml.calibration.positive_class_probability) - a calibrated probability if that label's
    classifier was calibrated, an uncalibrated classifier score otherwise. Earlier versions of
    this function used the classifier's hard predict() output instead, which for a binary 0.0/1.0
    target is always exactly 0.0 or 1.0 - not a graded score at all, and not what
    recommendation_ratio was ever meant to be compared against.

    Args:
        df_features (pandas.DataFrame): binary encoded features stored in a pandas dataframe
        model_dict (dict): dictionary with machine learning models with labels as keys

    Returns:
        list: email id per message, in the same order as df_features
        list: highest scoring label per message, in the same order
        numpy.ndarray: score of that highest scoring label per message, in the same order
        list[bool]: whether the highest scoring label's classifier was calibrated, in the same
            order - see mailsort.ml.calibration.is_calibrated()
    """
    df_predict = df_features.drop(["email_id"], axis=1)
    label_lst = list(model_dict.keys())
    proba_array = np.array(
        [
            positive_class_probability(model_dict[label], df_predict)
            for label in label_lst
        ]
    ).T
    calibrated_lst = [is_calibrated(model_dict[label]) for label in label_lst]
    argmax_indices = np.argmax(proba_array, axis=1)
    max_values = proba_array[np.arange(len(proba_array)), argmax_indices]
    best_label_lst = [label_lst[idx] for idx in argmax_indices]
    best_calibrated_lst = [calibrated_lst[idx] for idx in argmax_indices]
    return (
        df_features.email_id.tolist(),
        best_label_lst,
        max_values,
        best_calibrated_lst,
    )


def get_predictions_from_machine_learning_models(
    df_features: pandas.DataFrame,
    model_dict: dict[str, RandomForestClassifier],
    recommendation_ratio: float = 0.9,
) -> dict[str, str | None]:
    """
    Get recommendations from machine learning models

    Args:
        df_features (pandas.DataFrame): binary encoded features stored in a pandas dataframe
        model_dict (dict): dictionary with machine learning models with labels as keys
        recommendation_ratio (float): recommendation cutoff ratio

    Returns:
        dict: email id as keys and the corresponding newly assigned label as value
    """
    email_id_lst, best_label_lst, max_values, _ = _predict_top_label_per_message(
        df_features=df_features, model_dict=model_dict
    )
    new_label_lst = [
        label if max_val > recommendation_ratio else None
        for label, max_val in zip(best_label_lst, max_values, strict=False)
    ]
    return dict(zip(email_id_lst, new_label_lst, strict=False))


def score_messages_with_machine_learning_models(
    df_features: pandas.DataFrame,
    model_dict: dict[str, RandomForestClassifier],
    recommendation_ratio: float = 0.9,
) -> list[dict[str, Any]]:
    """
    Score every message in df_features against every per-label model in model_dict, without
    applying recommendation_ratio as a filter - unlike get_predictions_from_machine_learning_models,
    every message gets an entry in the result, whether or not its top label clears the threshold,
    so a caller can inspect *why* a message was or was not recommended (see
    mailsort.base.mail.AbstractMailBox.get_label_recommendations, the dry-run/preview API this
    was added for).

    This shares its underlying per-message, per-label scores with
    get_predictions_from_machine_learning_models (see _predict_top_label_per_message), so
    "threshold_reached" here always agrees with whether that function would recommend moving the
    same message for real, given the same recommendation_ratio.

    "score" is a positive-class probability (see mailsort.ml.calibration), calibrated if
    "calibrated" is True, an uncalibrated classifier score otherwise - the two are not
    interchangeable, see mailsort.ml.calibration for why, and mailsort.ml.evaluation to check
    whether a given recommendation_ratio behaves the way you expect on your own data.

    Args:
        df_features (pandas.DataFrame): binary encoded features stored in a pandas dataframe
        model_dict (dict): dictionary with machine learning models with labels as keys
        recommendation_ratio (float): recommendation cutoff ratio

    Returns:
        list: one dict per message in df_features, each with:
            - "email_id" (str): the message's email id
            - "recommended_label" (str/None): the label whose model scores this message highest,
              or None if model_dict is empty (e.g. no machine learning model trained yet)
            - "score" (float): that label's score for this message
            - "threshold_reached" (bool): whether "score" clears recommendation_ratio
            - "calibrated" (bool): whether "score" is a calibrated probability rather than a raw,
              uncalibrated classifier score - always False when model_dict is empty
    """
    if len(model_dict) == 0:
        return [
            {
                "email_id": email_id,
                "recommended_label": None,
                "score": 0.0,
                "threshold_reached": False,
                "calibrated": False,
            }
            for email_id in df_features.email_id.tolist()
        ]
    email_id_lst, best_label_lst, max_values, calibrated_lst = (
        _predict_top_label_per_message(df_features=df_features, model_dict=model_dict)
    )
    return [
        {
            "email_id": email_id,
            "recommended_label": label,
            "score": float(max_val),
            "threshold_reached": bool(max_val > recommendation_ratio),
            "calibrated": calibrated,
        }
        for email_id, label, max_val, calibrated in zip(
            email_id_lst, best_label_lst, max_values, calibrated_lst, strict=False
        )
    ]
