import os
import tempfile
from datetime import datetime
from unittest import TestCase

import pandas
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mailsort.base import get_email_database
from mailsort.status import DatabaseStatus, get_database_status
from mailsort.training import train_machine_learning_models


def _sample_emails_df():
    return pandas.DataFrame(
        [
            {
                "id": "id1",
                "from": "alice@example.com",
                "to": ["me@example.com"],
                "cc": [],
                "labels": ["Inbox"],
                "threads": "t1",
                "subject": "Hello",
                "content": "Hi there",
                "date": datetime(2024, 1, 1),
            },
            {
                "id": "id2",
                "from": "bob@example.com",
                "to": ["me@example.com"],
                "cc": [],
                "labels": ["Receipts"],
                "threads": "t2",
                "subject": "Your invoice",
                "content": "Invoice attached",
                "date": datetime(2024, 1, 2),
            },
        ]
    )


class TrainingAndStatusIntegrationTest(TestCase):
    """
    mailsort.training and mailsort.status open the database directly from a connection
    string (rather than through mailsort.Imap), so they need a file-backed sqlite database
    here to see data written by an earlier, independent connection - unlike sqlite:///:memory:,
    which is a fresh isolated database per connection.
    """

    def setUp(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(os.remove, path)
        self.connection_str = f"sqlite:///{path}"

        engine = create_engine(self.connection_str)
        session = sessionmaker(bind=engine)()
        db_email = get_email_database(engine=engine, session=session)
        db_email.store_dataframe(df=_sample_emails_df(), user_id=1)
        session.close()

    def test_train_then_status_reports_consistent_state(self):
        model_count = train_machine_learning_models(
            connection_str=self.connection_str,
            db_user_id=1,
            n_estimators=5,
            max_features=2,
            max_workers=1,
        )
        self.assertEqual(model_count, 2)

        status = get_database_status(connection_str=self.connection_str, db_user_id=1)
        self.assertIsInstance(status, DatabaseStatus)
        self.assertEqual(status.connection_str, self.connection_str)
        self.assertEqual(status.message_count, 2)
        self.assertEqual(status.active_message_count, 2)
        self.assertEqual(status.deleted_message_count, 0)
        self.assertEqual(set(status.trained_label_lst), {"Inbox", "Receipts"})
        self.assertGreater(status.feature_count, 0)

    def test_status_before_training_reports_no_models(self):
        status = get_database_status(connection_str=self.connection_str, db_user_id=1)

        self.assertEqual(status.message_count, 2)
        self.assertEqual(status.trained_label_lst, [])
        self.assertEqual(status.feature_count, 0)

    def test_status_is_scoped_by_db_user_id(self):
        status = get_database_status(connection_str=self.connection_str, db_user_id=2)

        self.assertEqual(status.message_count, 0)
        self.assertEqual(status.active_message_count, 0)


if __name__ == "__main__":
    import unittest

    unittest.main()
