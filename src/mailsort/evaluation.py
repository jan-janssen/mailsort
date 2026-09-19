"""
Reusable evaluation entry point for the ``mailsort evaluate`` CLI command.

Like mailsort.training, evaluation only touches the local database (see
mailsort.ml.evaluation for what it actually does - trains and scores a throwaway held-out split,
never the production models), so this module opens the database directly instead of going through
a mail backend such as mailsort.Imap, which would require live mail server credentials it does
not need.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mailsort.base import get_email_database
from mailsort.ml.evaluation import (
    EvaluationReport,
    FolderMetrics,
    HoldoutScores,
    evaluate_at_threshold,
    evaluate_machine_learning_models,
    evaluate_thresholds,
    score_holdout_split,
)

__all__ = [
    "EvaluationReport",
    "FolderMetrics",
    "HoldoutScores",
    "evaluate_at_threshold",
    "evaluate_models",
    "evaluate_thresholds",
    "score_holdout",
]


def evaluate_models(
    connection_str,
    db_user_id=1,
    recommendation_ratio=0.9,
    test_size=0.25,
    random_state=42,
    n_estimators=100,
    max_features=400,
    bootstrap=True,
    include_deleted=False,
    calibrate=True,
    min_samples_per_class_for_calibration=20,
    max_calibration_cv_folds=5,
    max_workers=None,
    label_prefix="labels_",
):
    """
    Evaluate the machine learning training procedure on the emails already stored in the local
    database, at a single `recommendation_ratio` - see mailsort.ml.evaluation for what this does
    and does not tell you, and why. This never modifies the production models stored in the
    database (see mailsort.training.train_machine_learning_models for that).

    Args:
        connection_str (str): SQLAlchemy connection string, e.g. "sqlite:///email.db"
        db_user_id (int): database user id - default: 1
        recommendation_ratio (float): cutoff a score must strictly clear to be "accepted"
        test_size (float): fraction of email threads held out for testing (0<test_size<1)
        random_state (int): random seed for the split and the classifiers, for reproducibility
        n_estimators (int): number of estimators per classifier
        max_features (int): maximum number of features per classifier
        bootstrap (bool): whether bootstrap samples are used when building trees
        include_deleted (bool): include emails marked as deleted - default: False
        calibrate (bool): calibrate each folder's classifier when there is enough data to do so
            safely - default: True. See mailsort.ml.calibration.
        min_samples_per_class_for_calibration (int): see
            mailsort.ml.calibration.should_calibrate()
        max_calibration_cv_folds (int): see mailsort.ml.calibration.calibration_cv_folds()
        max_workers (int): maximum number of workers for the machine learning models
        label_prefix (str): prefix used to recognise label columns during feature encoding

    Returns:
        mailsort.ml.evaluation.EvaluationReport
    """
    engine = create_engine(connection_str)
    session = sessionmaker(bind=engine)()
    try:
        db_email = get_email_database(engine=engine, session=session)
        df_all = db_email.get_all_emails(
            include_deleted=include_deleted, user_id=db_user_id
        )
        return evaluate_machine_learning_models(
            df_all,
            recommendation_ratio=recommendation_ratio,
            test_size=test_size,
            random_state=random_state,
            n_estimators=n_estimators,
            max_features=max_features,
            bootstrap=bootstrap,
            calibrate=calibrate,
            min_samples_per_class_for_calibration=min_samples_per_class_for_calibration,
            max_calibration_cv_folds=max_calibration_cv_folds,
            max_workers=max_workers,
            label_prefix=label_prefix,
        )
    finally:
        session.close()
        # Without this, pooled DBAPI connections stay open after session.close() alone,
        # which on Windows keeps the sqlite file locked for anyone trying to open or
        # delete it right after this function returns.
        engine.dispose()


def score_holdout(
    connection_str,
    db_user_id=1,
    test_size=0.25,
    random_state=42,
    n_estimators=100,
    max_features=400,
    bootstrap=True,
    include_deleted=False,
    calibrate=True,
    min_samples_per_class_for_calibration=20,
    max_calibration_cv_folds=5,
    max_workers=None,
    label_prefix="labels_",
):
    """
    Connection-string equivalent of mailsort.ml.evaluation.score_holdout_split() - trains and
    scores a held-out split once, so evaluate_at_threshold() can be called repeatedly afterwards
    to compare recommendation_ratio choices without retraining. See evaluate_models() to get a
    single EvaluationReport directly instead.

    Args: see evaluate_models() (same meaning, minus recommendation_ratio)

    Returns:
        mailsort.ml.evaluation.HoldoutScores
    """
    engine = create_engine(connection_str)
    session = sessionmaker(bind=engine)()
    try:
        db_email = get_email_database(engine=engine, session=session)
        df_all = db_email.get_all_emails(
            include_deleted=include_deleted, user_id=db_user_id
        )
        return score_holdout_split(
            df_all,
            test_size=test_size,
            random_state=random_state,
            n_estimators=n_estimators,
            max_features=max_features,
            bootstrap=bootstrap,
            calibrate=calibrate,
            min_samples_per_class_for_calibration=min_samples_per_class_for_calibration,
            max_calibration_cv_folds=max_calibration_cv_folds,
            max_workers=max_workers,
            label_prefix=label_prefix,
        )
    finally:
        session.close()
        engine.dispose()
