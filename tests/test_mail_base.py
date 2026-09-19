from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from mailsort.base.mail import AbstractMailBox
from mailsort.results import Prediction, ScoreType, SortResult, SyncResult, TrainResult


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


def _make_prediction(
    message_id,
    source_folder="Inbox",
    recommended_folder=None,
    score=0.0,
    threshold=0.9,
    accepted=False,
    score_type=ScoreType.RAW,
    subject=None,
):
    """Shorthand for building a Prediction fixture in tests that only care about a few fields."""
    return Prediction(
        message_id=message_id,
        source_folder=source_folder,
        recommended_folder=recommended_folder,
        score=score,
        threshold=threshold,
        accepted=accepted,
        score_type=score_type,
        subject=subject,
    )


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
        prediction_lst = [
            _make_prediction("id1", recommended_folder=None, accepted=True),
            _make_prediction("id2", recommended_folder="Inbox", accepted=True),
            _make_prediction("id3", recommended_folder="Spam", accepted=True),
        ]

        moved_lst = mailbox._move_emails(
            prediction_lst=prediction_lst,
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

    @patch("mailsort.base.mail.score_messages_with_machine_learning_models")
    @patch("mailsort.base.mail.encode_df_for_machine_learning")
    def test_filter_messages_from_server_moves_accepted_predictions(
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
                "calibrated": True,
            }
        ]

        result = mailbox.filter_messages_from_server(label="Inbox")

        self.assertEqual(mailbox.modify_calls, [("id1", ["Inbox"], ["Sorted"])])
        self.assertEqual(result, SortResult([("id1", "Sorted")]))
        self.assertEqual(result.moved_count, 1)

    @patch("mailsort.base.mail.score_messages_with_machine_learning_models")
    @patch("mailsort.base.mail.encode_df_for_machine_learning")
    def test_filter_messages_from_server_never_moves_abstained_predictions(
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
        # score below the (default 0.9) threshold - the model has an opinion, but not a
        # confident enough one, so the prediction abstains and must not be moved
        score_mock.return_value = [
            {
                "email_id": "id1",
                "recommended_label": "Sorted",
                "score": 0.4,
                "threshold_reached": False,
                "calibrated": False,
            }
        ]

        result = mailbox.filter_messages_from_server(label="Inbox")

        self.assertEqual(mailbox.modify_calls, [])
        self.assertEqual(result, SortResult([]))

    def test_filter_messages_from_server_without_trained_models_does_not_crash(self):
        # Regression test: get_label_recommendations() has always special-cased an untrained
        # database (no per-label models yet) to abstain without running the ML pipeline.
        # filter_messages_from_server() now shares that same code path via
        # get_label_recommendations(), so it inherits the same safety - previously it called the
        # ML pipeline unconditionally and crashed on an untrained database instead of abstaining.
        db_ml = MagicMock()
        db_ml.load_models.return_value = ({}, [])
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

        result = mailbox.filter_messages_from_server(label="Inbox")

        self.assertEqual(mailbox.modify_calls, [])
        self.assertEqual(result, SortResult([]))

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

    def test_get_label_recommendations_without_trained_models_abstains_for_every_message(
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

        predictions = mailbox.get_label_recommendations(
            label="Inbox", recommendation_ratio=0.9
        )

        self.assertEqual(
            predictions,
            [
                Prediction(
                    message_id="id1",
                    source_folder="Inbox",
                    recommended_folder=None,
                    score=0.0,
                    threshold=0.9,
                    accepted=False,
                    score_type=ScoreType.RAW,
                    subject="Hello",
                ),
                Prediction(
                    message_id="id2",
                    source_folder="Inbox",
                    recommended_folder=None,
                    score=0.0,
                    threshold=0.9,
                    accepted=False,
                    score_type=ScoreType.RAW,
                    subject="World",
                ),
            ],
        )
        self.assertEqual(mailbox.modify_calls, [])

    @patch("mailsort.base.mail.score_messages_with_machine_learning_models")
    @patch("mailsort.base.mail.encode_df_for_machine_learning")
    def test_get_label_recommendations_wires_scores_into_an_accepted_prediction(
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
                "calibrated": True,
            }
        ]

        predictions = mailbox.get_label_recommendations(
            label="Inbox", recommendation_ratio=0.9
        )

        self.assertEqual(
            predictions,
            [
                Prediction(
                    message_id="id1",
                    source_folder="Inbox",
                    recommended_folder="Sorted",
                    score=0.95,
                    threshold=0.9,
                    accepted=True,
                    score_type=ScoreType.CALIBRATED,
                    subject="Hello",
                )
            ],
        )
        score_mock.assert_called_once()
        self.assertEqual(score_mock.call_args.kwargs["recommendation_ratio"], 0.9)
        self.assertEqual(mailbox.modify_calls, [])

    @patch("mailsort.base.mail.score_messages_with_machine_learning_models")
    @patch("mailsort.base.mail.encode_df_for_machine_learning")
    def test_get_label_recommendations_below_threshold_abstains(
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
        # the model still names a recommended_folder - it simply is not confident enough
        score_mock.return_value = [
            {
                "email_id": "id1",
                "recommended_label": "Sorted",
                "score": 0.5,
                "threshold_reached": False,
                "calibrated": False,
            }
        ]

        (prediction,) = mailbox.get_label_recommendations(
            label="Inbox", recommendation_ratio=0.9
        )

        self.assertEqual(prediction.recommended_folder, "Sorted")
        self.assertEqual(prediction.score, 0.5)
        self.assertEqual(prediction.threshold, 0.9)
        self.assertFalse(prediction.accepted)
        self.assertEqual(prediction.score_type, ScoreType.RAW)

    @patch("mailsort.base.mail.score_messages_with_machine_learning_models")
    @patch("mailsort.base.mail.encode_df_for_machine_learning")
    def test_get_label_recommendations_score_exactly_at_threshold_abstains(
        self, encode_mock, score_mock
    ):
        # Boundary/tie case: a score exactly equal to the threshold must not be accepted - the
        # cutoff is a strict ">", mirrored here from score_messages_with_machine_learning_models.
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
                "score": 0.9,
                # score == recommendation_ratio - a real score_messages_with_machine_learning_models()
                # call would compute this as False too, since the cutoff is a strict ">"
                "threshold_reached": False,
                "calibrated": False,
            }
        ]

        (prediction,) = mailbox.get_label_recommendations(
            label="Inbox", recommendation_ratio=0.9
        )

        self.assertEqual(prediction.score, prediction.threshold)
        self.assertFalse(prediction.accepted)

    @patch("mailsort.base.mail.score_messages_with_machine_learning_models")
    @patch("mailsort.base.mail.encode_df_for_machine_learning")
    def test_get_label_recommendations_mixed_accepted_and_abstained(
        self, encode_mock, score_mock
    ):
        # Two trained folders effectively "tie" from the caller's point of view in that both
        # produce a recommendation for a message, but only the message whose top score clears
        # the threshold is accepted - the other abstains despite also having a recommended folder.
        db_ml = MagicMock()
        db_ml.load_models.return_value = (
            {"Sorted": MagicMock(), "Receipts": MagicMock()},
            ["f1"],
        )
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
                "subject": "Confident",
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
                "subject": "Unsure",
                "content": "c2",
                "date": None,
            },
        }
        encode_mock.return_value = pd.DataFrame(
            {"email_id": ["id1", "id2"], "f1": [1, 0]}
        )
        score_mock.return_value = [
            {
                "email_id": "id1",
                "recommended_label": "Sorted",
                "score": 0.95,
                "threshold_reached": True,
                "calibrated": True,
            },
            {
                "email_id": "id2",
                "recommended_label": "Receipts",
                "score": 0.3,
                "threshold_reached": False,
                "calibrated": False,
            },
        ]

        predictions = mailbox.get_label_recommendations(
            label="Inbox", recommendation_ratio=0.9
        )

        accepted = [p for p in predictions if p.accepted]
        abstained = [p for p in predictions if not p.accepted]
        self.assertEqual([p.message_id for p in accepted], ["id1"])
        self.assertEqual([p.message_id for p in abstained], ["id2"])
        self.assertEqual(abstained[0].recommended_folder, "Receipts")
