"""
Reusable status reporting for the ``mailsort status`` CLI command.

Like mailsort.training, this reads the local database directly rather than going through a
mail backend such as mailsort.Imap, since reporting on the local database does not need a live
connection to the mail server.
"""

from dataclasses import dataclass

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mailsort.base import get_email_database
from mailsort.ml import get_machine_learning_database


@dataclass
class DatabaseStatus:
    connection_str: str
    db_user_id: int
    message_count: int
    active_message_count: int
    deleted_message_count: int
    trained_label_lst: list
    feature_count: int


def get_database_status(connection_str, db_user_id=1):
    """
    Summarize the local mailsort database: where it is, how many messages it knows about, and
    how many per-folder machine learning models have been trained - without connecting to any
    mail server.

    Args:
        connection_str (str): SQLAlchemy connection string, e.g. "sqlite:///email.db"
        db_user_id (int): database user id - default: 1

    Returns:
        DatabaseStatus
    """
    engine = create_engine(connection_str)
    session = sessionmaker(bind=engine)()
    try:
        db_email = get_email_database(engine=engine, session=session)
        db_ml = get_machine_learning_database(engine=engine, session=session)
        active_message_count = db_email.count_emails(
            include_deleted=False, user_id=db_user_id
        )
        message_count = db_email.count_emails(include_deleted=True, user_id=db_user_id)
        trained_label_lst = sorted(db_ml.get_labels(user_id=db_user_id))
        feature_count = len(db_ml.get_features(user_id=db_user_id))
        return DatabaseStatus(
            connection_str=connection_str,
            db_user_id=db_user_id,
            message_count=message_count,
            active_message_count=active_message_count,
            deleted_message_count=message_count - active_message_count,
            trained_label_lst=trained_label_lst,
            feature_count=feature_count,
        )
    finally:
        session.close()
