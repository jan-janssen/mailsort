"""
Reusable training entry point for the ``mailsort train`` CLI command.

Training only touches the local database (see
mailsort.base.mail.AbstractMailBox.fit_machine_learning_model_to_database for the equivalent
step in the fetch-store-train-predict-move loop), so this module opens the database directly
instead of going through a mail backend such as mailsort.Imap, which would require live mail
server credentials it does not need.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mailsort.base import get_email_database
from mailsort.ml import (
    DEFAULT_MAX_CALIBRATION_CV_FOLDS,
    DEFAULT_MIN_SAMPLES_PER_CLASS_FOR_CALIBRATION,
    encode_df_for_machine_learning,
    fit_machine_learning_models,
    get_machine_learning_database,
)


def train_machine_learning_models(
    connection_str,
    db_user_id=1,
    n_estimators=100,
    max_features=400,
    random_state=42,
    bootstrap=True,
    include_deleted=False,
    max_workers=None,
    calibrate=True,
    min_samples_per_class_for_calibration=DEFAULT_MIN_SAMPLES_PER_CLASS_FOR_CALIBRATION,
    max_calibration_cv_folds=DEFAULT_MAX_CALIBRATION_CV_FOLDS,
):
    """
    Train one machine learning model per folder on the emails already stored in the local
    database, and store the result back in the same database.

    Args:
        connection_str (str): SQLAlchemy connection string, e.g. "sqlite:///email.db"
        db_user_id (int): database user id - default: 1
        n_estimators (int): number of estimators
        max_features (int): number of features
        random_state (int): random state
        bootstrap (bool): whether bootstrap samples are used when building trees - default: True
        include_deleted (bool): include emails marked as deleted - default: False
        max_workers (int): maximum number of workers for the machine learning models
        calibrate (bool): calibrate each folder's classifier when there is enough data to do so
            safely - default: True. See mailsort.ml.calibration for what this means and why it
            is conditional rather than automatic; see mailsort.ml.evaluation to check whether the
            resulting scores behave the way you expect on your own data.
        min_samples_per_class_for_calibration (int): see
            mailsort.ml.calibration.should_calibrate()
        max_calibration_cv_folds (int): see mailsort.ml.calibration.calibration_cv_folds()

    Returns:
        int: number of per-folder models trained
    """
    engine = create_engine(connection_str)
    session = sessionmaker(bind=engine)()
    try:
        db_email = get_email_database(engine=engine, session=session)
        db_ml = get_machine_learning_database(engine=engine, session=session)
        df_all = db_email.get_all_emails(
            include_deleted=include_deleted, user_id=db_user_id
        )
        df_all_features, df_all_labels = encode_df_for_machine_learning(
            df=df_all, feature_lst=[], label_lst=[], return_labels=True
        )
        df_all_features = df_all_features.loc[
            :, ~df_all_features.columns.duplicated()
        ].copy()
        df_all_features = df_all_features.reindex(
            sorted(df_all_features.columns), axis=1
        )
        model_dict = fit_machine_learning_models(
            df_all_features=df_all_features,
            df_all_labels=df_all_labels,
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_state,
            bootstrap=bootstrap,
            max_workers=max_workers,
            calibrate=calibrate,
            min_samples_per_class_for_calibration=min_samples_per_class_for_calibration,
            max_calibration_cv_folds=max_calibration_cv_folds,
        )
        db_ml.store_models(
            model_dict=model_dict,
            feature_lst=df_all_features.columns.values.tolist(),
            user_id=db_user_id,
            commit=True,
        )
        return len(model_dict)
    finally:
        session.close()
        # Without this, pooled DBAPI connections stay open after session.close() alone,
        # which on Windows keeps the sqlite file locked for anyone trying to open or
        # delete it right after this function returns.
        engine.dispose()
