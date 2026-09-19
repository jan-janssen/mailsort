from unittest import TestCase
from unittest.mock import create_autospec

from mailsort.base.mail import AbstractMailBox
from mailsort.results import Prediction, SortResult, SyncResult, TrainResult
from mailsort.sorter import MailSorter


class MailSorterTest(TestCase):
    """
    MailSorter is a thin facade over AbstractMailBox, so these tests only check wiring - that
    each method calls the right underlying AbstractMailBox method with the right arguments and
    returns/wraps its result - not the underlying fetch-store-train-predict-move logic itself,
    which is already covered by tests/test_mail_base.py.

    The mailbox double is built with create_autospec(AbstractMailBox, instance=True) rather than
    a plain MagicMock(), so a call with a keyword argument AbstractMailBox does not actually
    accept would fail the test instead of silently succeeding.
    """

    def setUp(self):
        self.mailbox = create_autospec(AbstractMailBox, instance=True)
        self.sorter = MailSorter(self.mailbox)

    def test_sync_delegates_to_update_database_with_defaults(self):
        self.mailbox.update_database.return_value = SyncResult(
            new_message_count=0, updated_message_count=0, deleted_message_count=0
        )

        result = self.sorter.sync()

        self.mailbox.update_database.assert_called_once_with(
            quick=False, label_lst=None, email_format=None
        )
        self.assertEqual(
            result,
            SyncResult(
                new_message_count=0, updated_message_count=0, deleted_message_count=0
            ),
        )

    def test_sync_forwards_custom_arguments(self):
        self.mailbox.update_database.return_value = SyncResult(
            new_message_count=3, updated_message_count=1, deleted_message_count=2
        )

        result = self.sorter.sync(quick=True, label_lst=["Inbox"], email_format="full")

        self.mailbox.update_database.assert_called_once_with(
            quick=True, label_lst=["Inbox"], email_format="full"
        )
        self.assertEqual(result.new_message_count, 3)

    def test_train_delegates_to_fit_machine_learning_model_to_database(self):
        self.mailbox.fit_machine_learning_model_to_database.return_value = TrainResult(
            trained_label_lst=["Inbox", "Receipts"]
        )

        result = self.sorter.train(
            n_estimators=5, max_features=2, include_deleted=True, max_workers=1
        )

        self.mailbox.fit_machine_learning_model_to_database.assert_called_once_with(
            n_estimators=5,
            max_features=2,
            random_state=42,
            bootstrap=True,
            include_deleted=True,
            max_workers=1,
        )
        self.assertEqual(result.trained_label_lst, ["Inbox", "Receipts"])
        self.assertEqual(result.model_count, 2)

    def test_predict_wraps_recommendation_dicts_into_predictions(self):
        self.mailbox.get_label_recommendations.return_value = [
            {
                "message_id": "MailSortInbox\x1f1",
                "subject": "Hello",
                "recommended_label": "Sorted",
                "score": 0.95,
                "threshold_reached": True,
            },
            {
                "message_id": "MailSortInbox\x1f2",
                "subject": None,
                "recommended_label": None,
                "score": 0.0,
                "threshold_reached": False,
            },
        ]

        predictions = self.sorter.predict(
            "MailSortInbox", recommendation_ratio=0.8, label_prefix="custom_"
        )

        self.mailbox.get_label_recommendations.assert_called_once_with(
            label="MailSortInbox", recommendation_ratio=0.8, label_prefix="custom_"
        )
        self.assertEqual(
            predictions,
            [
                Prediction(
                    message_id="MailSortInbox\x1f1",
                    subject="Hello",
                    recommended_label="Sorted",
                    score=0.95,
                    threshold_reached=True,
                ),
                Prediction(
                    message_id="MailSortInbox\x1f2",
                    subject=None,
                    recommended_label=None,
                    score=0.0,
                    threshold_reached=False,
                ),
            ],
        )

    def test_predict_never_calls_sort(self):
        self.mailbox.get_label_recommendations.return_value = []

        self.sorter.predict("MailSortInbox")

        self.mailbox.filter_messages_from_server.assert_not_called()

    def test_predict_empty_folder_returns_empty_list(self):
        self.mailbox.get_label_recommendations.return_value = []

        self.assertEqual(self.sorter.predict("Empty"), [])

    def test_sort_delegates_to_filter_messages_from_server(self):
        self.mailbox.filter_messages_from_server.return_value = SortResult(
            moved_lst=[("MailSortInbox\x1f1", "Sorted")]
        )

        result = self.sorter.sort(
            "MailSortInbox", recommendation_ratio=0.7, label_prefix="custom_"
        )

        self.mailbox.filter_messages_from_server.assert_called_once_with(
            label="MailSortInbox", recommendation_ratio=0.7, label_prefix="custom_"
        )
        self.assertEqual(result.moved_count, 1)
        self.assertEqual(result.moved_lst, [("MailSortInbox\x1f1", "Sorted")])

    def test_close_delegates_to_mailbox(self):
        self.sorter.close()

        self.mailbox.close.assert_called_once_with()

    def test_context_manager_closes_mailbox_on_exit(self):
        with MailSorter(self.mailbox) as sorter:
            self.assertIsInstance(sorter, MailSorter)
            self.mailbox.close.assert_not_called()

        self.mailbox.close.assert_called_once_with()

    def test_context_manager_closes_mailbox_even_on_exception(self):
        with self.assertRaises(ValueError), MailSorter(self.mailbox):
            raise ValueError("boom")

        self.mailbox.close.assert_called_once_with()

    def test_mailbox_property_exposes_wrapped_backend(self):
        self.assertIs(self.sorter.mailbox, self.mailbox)


if __name__ == "__main__":
    import unittest

    unittest.main()
