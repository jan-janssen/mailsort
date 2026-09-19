from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from mailsort.base.mail import AbstractMailBox
from mailsort.results import SortResult, SyncResult, TrainResult


class _StubMailBox(AbstractMailBox):
    """Minimal concrete AbstractMailBox used to test the shared loop in isolation."""

    def __init__(self, label_dict_fixture=None, **kwargs):
        self.label_dict_fixture = label_dict_fixture or {
            "Inbox": "Inbox",
            "Spam": "Spam",
        }
        self.search_result = []
        self.message_detail_dict = {}
        self.modify_calls = []
        self.labels_for_email_dict = {}
        super().__init__(mail_service=MagicMock(), **kwargs)

    def _search_email_on_server(
        self, query_string="", label_lst=None, only_message_ids=False
    ):
        return self.search_result

    def _get_message_detail(self, message_id, email_format=None, metadata_headers=None):
        return self.message_detail_dict.get(message_id)

    def _get_label_translate_dict(self):
        return self.label_dict_fixture

    def _modify_message_labels(
        self, message_id, label_id_remove_lst=None, label_id_add_lst=None
    ):
        self.modify_calls.append((message_id, label_id_remove_lst, label_id_add_lst))

    def _get_labels_for_email(self, message_id):
        return self.labels_for_email_dict.get(message_id, [])

    def _parse_message(self, message):
        return message


class AbstractMailBoxTest(TestCase):
    def test_labels_property(self):
        mailbox = _StubMailBox()
        self.assertEqual(sorted(mailbox.labels), ["Inbox", "Spam"])

    def test_close_default_implementation_is_a_no_op(self):
        mailbox = _StubMailBox()

        mailbox.close()  # must not raise, even though _StubMailBox never overrides it

    def test_context_manager_calls_close_and_propagates_exceptions(self):
        mailbox = _StubMailBox()
        mailbox.close = MagicMock()

        with self.assertRaises(ValueError), mailbox as entered:
            self.assertIs(entered, mailbox)
            raise ValueError("boom")

        mailbox.close.assert_called_once_with()

    def test_download_emails_for_label(self):
        mailbox = _StubMailBox()
        mailbox.search_result = ["id1", "id2"]
        mailbox.message_detail_dict = {
            "id1": {
                "id": "id1",
                "threads": "t1",
                "labels": [],
                "to": [],
                "from": None,
                "cc": [],
                "subject": "s1",
                "content": "c1",
                "date": None,
            },
            "id2": None,
        }

        df = mailbox.download_emails_for_label(label="Inbox")

        self.assertEqual(df["id"].tolist(), ["id1"])

    def test_move_emails_skips_matching_or_none_labels(self):
        mailbox = _StubMailBox()

        moved_lst = mailbox._move_emails(
            move_email_dict={"id1": None, "id2": "Inbox", "id3": "Spam"},
            label_to_ignore="Inbox",
        )

        self.assertEqual(mailbox.modify_calls, [("id3", ["Inbox"], ["Spam"])])
        self.assertEqual(moved_lst, [("id3", "Spam")])

    def test_update_database_marks_missing_as_deleted(self):
        db_email = MagicMock()
        db_email.get_labels_to_update.return_value = (["new"], [], ["deleted"])
        mailbox = _StubMailBox(database_email=db_email)
        mailbox.search_result = ["new"]
        mailbox.message_detail_dict = {
            "new": {
                "id": "new",
                "threads": "t",
                "labels": [],
                "to": [],
                "from": None,
                "cc": [],
                "subject": "s",
                "content": "c",
                "date": None,
            }
        }

        result = mailbox.update_database(quick=False)

        db_email.mark_emails_as_deleted.assert_called_once_with(
            message_id_lst=["deleted"], user_id=1
        )
        db_email.store_dataframe.assert_called_once()
        self.assertEqual(result, SyncResult(1, 0, 1))

    def test_update_database_quick_skips_relabel_and_delete_counts(self):
        db_email = MagicMock()
        db_email.get_labels_to_update.return_value = (["new"], ["updated"], ["deleted"])
        mailbox = _StubMailBox(database_email=db_email)
        mailbox.search_result = ["new"]
        mailbox.message_detail_dict = {
            "new": {
                "id": "new",
                "threads": "t",
                "labels": [],
                "to": [],
                "from": None,
                "cc": [],
                "subject": "s",
                "content": "c",
                "date": None,
            }
        }

        result = mailbox.update_database(quick=True)

        db_email.mark_emails_as_deleted.assert_not_called()
        db_email.update_labels.assert_not_called()
        self.assertEqual(result, SyncResult(1, 0, 0))

    def test_update_database_without_email_database_is_a_no_op(self):
        mailbox = _StubMailBox()

        result = mailbox.update_database(quick=False)

        self.assertEqual(result, SyncResult(0, 0, 0))

    @patch("mailsort.base.mail.fit_machine_learning_models")
    @patch("mailsort.base.mail.encode_df_for_machine_learning")
    def test_fit_machine_learning_model_to_database(self, encode_mock, fit_mock):
        db_email = MagicMock()
        db_email.get_all_emails.return_value = pd.DataFrame(
            [
                {
                    "id": "x",
                    "from": "a@b.com",
                    "to": [],
                    "cc": [],
                    "labels": [],
                    "threads": "t",
                }
            ]
        )
        db_ml = MagicMock()
        mailbox = _StubMailBox(database_email=db_email, database_ml=db_ml)
        features = pd.DataFrame([{"email_id": "x", "f1": 1}])
        labels = pd.DataFrame([{"labels_Inbox": 1}])
        encode_mock.return_value = (features, labels)
        fit_mock.return_value = {"Inbox": MagicMock()}

        result = mailbox.fit_machine_learning_model_to_database(
            n_estimators=5, max_features=2
        )

        db_ml.store_models.assert_called_once()
        self.assertEqual(result, TrainResult(["Inbox"]))
        self.assertEqual(result.model_count, 1)

    @patch("mailsort.base.mail.get_predictions_from_machine_learning_models")
    @patch("mailsort.base.mail.encode_df_for_machine_learning")
    def test_filter_messages_from_server_moves_recommended_messages(
        self, encode_mock, predict_mock
    ):
        db_ml = MagicMock()
        db_ml.load_models.return_value = ({"Sorted": MagicMock()}, ["f1"])
        mailbox = _StubMailBox(database_ml=db_ml)
        mailbox.search_result = ["id1"]
        mailbox.message_detail_dict = {
            "id1": {
                "id": "id1",
                "threads": "t1",
                "labels": ["Inbox"],
                "to": [],
                "from": None,
                "cc": [],
                "subject": "Hello",
                "content": "c1",
                "date": None,
            }
        }
        encode_mock.return_value = pd.DataFrame({"email_id": ["id1"], "f1": [1]})
        predict_mock.return_value = {"id1": "Sorted"}

        result = mailbox.filter_messages_from_server(label="Inbox")

        self.assertEqual(mailbox.modify_calls, [("id1", ["Inbox"], ["Sorted"])])
        self.assertEqual(result, SortResult([("id1", "Sorted")]))
        self.assertEqual(result.moved_count, 1)

    def test_filter_messages_from_server_empty_folder_returns_empty_sort_result(self):
        mailbox = _StubMailBox()
        mailbox.search_result = []

        result = mailbox.filter_messages_from_server(label="Inbox")

        self.assertEqual(result, SortResult([]))
        self.assertEqual(result.moved_count, 0)

    def test_get_label_recommendations_returns_empty_list_for_empty_folder(self):
        db_ml = MagicMock()
        mailbox = _StubMailBox(database_ml=db_ml)
        mailbox.search_result = []

        recommendations = mailbox.get_label_recommendations(label="Inbox")

        self.assertEqual(recommendations, [])
        db_ml.load_models.assert_not_called()
        self.assertEqual(mailbox.modify_calls, [])

    def test_get_label_recommendations_without_trained_models_never_moves_anything(
        self,
    ):
        db_ml = MagicMock()
        db_ml.load_models.return_value = ({}, [])
        mailbox = _StubMailBox(database_ml=db_ml)
        mailbox.search_result = ["id1", "id2"]
        mailbox.message_detail_dict = {
            "id1": {
                "id": "id1",
                "threads": "t1",
                "labels": ["Inbox"],
                "to": [],
                "from": None,
                "cc": [],
                "subject": "Hello",
                "content": "c1",
                "date": None,
            },
            "id2": {
                "id": "id2",
                "threads": "t2",
                "labels": ["Inbox"],
                "to": [],
                "from": None,
                "cc": [],
                "subject": "World",
                "content": "c2",
                "date": None,
            },
        }

        recommendations = mailbox.get_label_recommendations(label="Inbox")

        self.assertEqual(
            [(entry["message_id"], entry["subject"]) for entry in recommendations],
            [("id1", "Hello"), ("id2", "World")],
        )
        self.assertTrue(
            all(entry["recommended_label"] is None for entry in recommendations)
        )
        self.assertTrue(
            all(not entry["threshold_reached"] for entry in recommendations)
        )
        self.assertEqual(mailbox.modify_calls, [])

    @patch("mailsort.base.mail.score_messages_with_machine_learning_models")
    @patch("mailsort.base.mail.encode_df_for_machine_learning")
    def test_get_label_recommendations_wires_scores_and_never_moves(
        self, encode_mock, score_mock
    ):
        db_ml = MagicMock()
        db_ml.load_models.return_value = ({"Sorted": MagicMock()}, ["f1"])
        mailbox = _StubMailBox(database_ml=db_ml)
        mailbox.search_result = ["id1"]
        mailbox.message_detail_dict = {
            "id1": {
                "id": "id1",
                "threads": "t1",
                "labels": ["Inbox"],
                "to": [],
                "from": None,
                "cc": [],
                "subject": "Hello",
                "content": "c1",
                "date": None,
            }
        }
        encode_mock.return_value = pd.DataFrame({"email_id": ["id1"], "f1": [1]})
        score_mock.return_value = [
            {
                "email_id": "id1",
                "recommended_label": "Sorted",
                "score": 0.95,
                "threshold_reached": True,
            }
        ]

        recommendations = mailbox.get_label_recommendations(
            label="Inbox", recommendation_ratio=0.9
        )

        self.assertEqual(
            recommendations,
            [
                {
                    "message_id": "id1",
                    "subject": "Hello",
                    "recommended_label": "Sorted",
                    "score": 0.95,
                    "threshold_reached": True,
                }
            ],
        )
        score_mock.assert_called_once()
        self.assertEqual(score_mock.call_args.kwargs["recommendation_ratio"], 0.9)
        self.assertEqual(mailbox.modify_calls, [])
