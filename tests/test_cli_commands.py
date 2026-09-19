from unittest import TestCase
from unittest.mock import patch

from mailsort.__main__ import (
    _EXIT_CONFIG_ERROR,
    _EXIT_OK,
    _EXIT_RUNTIME_ERROR,
    _format_status_table,
    command_line_parser,
)
from mailsort.status import DatabaseStatus


class SyncCommandTest(TestCase):
    @patch("mailsort.__main__.Imap")
    def test_sync_wires_imap_and_updates_database(self, imap_cls):
        imap_instance = imap_cls.return_value

        exit_code = command_line_parser(
            [
                "sync",
                "--host",
                "localhost",
                "--username",
                "user",
                "--password",
                "secret",
                "-d",
                "sqlite:///:memory:",
            ]
        )

        self.assertEqual(exit_code, _EXIT_OK)
        imap_cls.assert_called_once_with(
            host="localhost",
            port=993,
            username="user",
            password="secret",
            connection_str="sqlite:///:memory:",
            db_user_id=1,
            use_ssl=True,
            email_download_format="metadata",
        )
        imap_instance.update_database.assert_called_once_with(quick=False, label_lst=None)
        imap_instance.close.assert_called_once()

    @patch("mailsort.__main__.Imap")
    def test_sync_quick_and_folder_flags_are_forwarded(self, imap_cls):
        imap_instance = imap_cls.return_value

        exit_code = command_line_parser(
            [
                "sync",
                "--host",
                "localhost",
                "--username",
                "user",
                "--password",
                "secret",
                "--quick",
                "--folder",
                "Inbox",
                "--folder",
                "Archive",
            ]
        )

        self.assertEqual(exit_code, _EXIT_OK)
        imap_instance.update_database.assert_called_once_with(
            quick=True, label_lst=["Inbox", "Archive"]
        )

    @patch("mailsort.__main__.Imap")
    def test_sync_missing_connection_args_returns_config_error(self, imap_cls):
        exit_code = command_line_parser(["sync"])

        self.assertEqual(exit_code, _EXIT_CONFIG_ERROR)
        imap_cls.assert_not_called()

    @patch("mailsort.__main__.Imap")
    def test_sync_runtime_error_returns_runtime_error_code(self, imap_cls):
        imap_cls.return_value.update_database.side_effect = RuntimeError("boom")

        exit_code = command_line_parser(
            [
                "sync",
                "--host",
                "localhost",
                "--username",
                "user",
                "--password",
                "secret",
            ]
        )

        self.assertEqual(exit_code, _EXIT_RUNTIME_ERROR)


class TrainCommandTest(TestCase):
    @patch("mailsort.__main__.train_machine_learning_models")
    def test_train_calls_reusable_api_without_mail_connection(self, train_mock):
        train_mock.return_value = 3

        exit_code = command_line_parser(["train", "-d", "sqlite:///:memory:", "-i", "2"])

        self.assertEqual(exit_code, _EXIT_OK)
        train_mock.assert_called_once_with(
            connection_str="sqlite:///:memory:",
            db_user_id=2,
            include_deleted=False,
        )

    @patch("mailsort.__main__.train_machine_learning_models")
    def test_train_include_deleted_flag_is_forwarded(self, train_mock):
        train_mock.return_value = 0

        command_line_parser(["train", "--include-deleted"])

        train_mock.assert_called_once_with(
            connection_str="sqlite:///email.db",
            db_user_id=1,
            include_deleted=True,
        )

    @patch("mailsort.__main__.Imap")
    def test_train_does_not_require_host_or_credentials(self, imap_cls):
        with patch("mailsort.__main__.train_machine_learning_models", return_value=0):
            exit_code = command_line_parser(["train"])

        self.assertEqual(exit_code, _EXIT_OK)
        imap_cls.assert_not_called()

    @patch("mailsort.__main__.train_machine_learning_models")
    def test_train_runtime_error_returns_runtime_error_code(self, train_mock):
        train_mock.side_effect = RuntimeError("boom")

        exit_code = command_line_parser(["train"])

        self.assertEqual(exit_code, _EXIT_RUNTIME_ERROR)


class PredictCommandTest(TestCase):
    @patch("mailsort.__main__.Imap")
    def test_predict_prints_recommendations_and_never_moves_anything(self, imap_cls):
        imap_instance = imap_cls.return_value
        imap_instance.get_label_recommendations.return_value = [
            {
                "message_id": "MailSortInbox\x1f1",
                "subject": "Hello",
                "recommended_label": "Sorted",
                "score": 1.0,
                "threshold_reached": True,
            }
        ]

        with patch("builtins.print") as print_mock:
            exit_code = command_line_parser(
                [
                    "predict",
                    "MailSortInbox",
                    "--host",
                    "localhost",
                    "--username",
                    "user",
                    "--password",
                    "secret",
                ]
            )

        self.assertEqual(exit_code, _EXIT_OK)
        imap_instance.get_label_recommendations.assert_called_once_with(
            label="MailSortInbox", recommendation_ratio=0.9, label_prefix="labels_"
        )
        imap_instance.filter_messages_from_server.assert_not_called()
        printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list)
        self.assertIn("Hello", printed)
        self.assertIn("Sorted", printed)

    @patch("mailsort.__main__.Imap")
    def test_predict_forwards_custom_ratio_and_prefix(self, imap_cls):
        imap_instance = imap_cls.return_value
        imap_instance.get_label_recommendations.return_value = []

        command_line_parser(
            [
                "predict",
                "Inbox",
                "--host",
                "localhost",
                "--username",
                "user",
                "--password",
                "secret",
                "--recommendation-ratio",
                "0.5",
                "--label-prefix",
                "custom_",
            ]
        )

        imap_instance.get_label_recommendations.assert_called_once_with(
            label="Inbox", recommendation_ratio=0.5, label_prefix="custom_"
        )

    @patch("mailsort.__main__.Imap")
    def test_predict_missing_connection_args_returns_config_error(self, imap_cls):
        exit_code = command_line_parser(["predict", "Inbox"])

        self.assertEqual(exit_code, _EXIT_CONFIG_ERROR)
        imap_cls.assert_not_called()

    def test_predict_requires_folder_argument(self):
        with self.assertRaises(SystemExit) as context:
            command_line_parser(["predict"])
        self.assertEqual(context.exception.code, 2)


class SortCommandTest(TestCase):
    @patch("mailsort.__main__.Imap")
    def test_sort_moves_messages_via_filter(self, imap_cls):
        imap_instance = imap_cls.return_value

        exit_code = command_line_parser(
            [
                "sort",
                "MailSortInbox",
                "--host",
                "localhost",
                "--username",
                "user",
                "--password",
                "secret",
            ]
        )

        self.assertEqual(exit_code, _EXIT_OK)
        imap_instance.filter_messages_from_server.assert_called_once_with(
            label="MailSortInbox", recommendation_ratio=0.9, label_prefix="labels_"
        )
        imap_instance.close.assert_called_once()

    @patch("mailsort.__main__.Imap")
    def test_sort_missing_connection_args_returns_config_error(self, imap_cls):
        exit_code = command_line_parser(["sort", "Inbox"])

        self.assertEqual(exit_code, _EXIT_CONFIG_ERROR)
        imap_cls.assert_not_called()


class StatusCommandTest(TestCase):
    @patch("mailsort.__main__.get_database_status")
    def test_status_reports_database_summary(self, status_mock):
        status_mock.return_value = DatabaseStatus(
            connection_str="sqlite:///email.db",
            db_user_id=1,
            message_count=12,
            active_message_count=10,
            deleted_message_count=2,
            trained_label_lst=["Inbox", "Receipts"],
            feature_count=42,
        )

        with patch("builtins.print") as print_mock:
            exit_code = command_line_parser(["status"])

        self.assertEqual(exit_code, _EXIT_OK)
        status_mock.assert_called_once_with(
            connection_str="sqlite:///email.db", db_user_id=1
        )
        printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list)
        self.assertIn("sqlite:///email.db", printed)
        self.assertIn("12 total", printed)
        self.assertIn("Inbox", printed)
        self.assertIn("Receipts", printed)

    @patch("mailsort.__main__.get_database_status")
    def test_status_does_not_require_mail_credentials(self, status_mock):
        status_mock.return_value = DatabaseStatus(
            connection_str="sqlite:///email.db",
            db_user_id=1,
            message_count=0,
            active_message_count=0,
            deleted_message_count=0,
            trained_label_lst=[],
            feature_count=0,
        )

        exit_code = command_line_parser(["status", "-d", "sqlite:///other.db"])

        self.assertEqual(exit_code, _EXIT_OK)
        status_mock.assert_called_once_with(
            connection_str="sqlite:///other.db", db_user_id=1
        )

    @patch("mailsort.__main__.get_database_status")
    def test_status_runtime_error_returns_runtime_error_code(self, status_mock):
        status_mock.side_effect = RuntimeError("no such table")

        exit_code = command_line_parser(["status"])

        self.assertEqual(exit_code, _EXIT_RUNTIME_ERROR)


class FormatStatusTableTest(TestCase):
    def test_no_trained_labels(self):
        table = _format_status_table(
            DatabaseStatus(
                connection_str="sqlite:///email.db",
                db_user_id=1,
                message_count=0,
                active_message_count=0,
                deleted_message_count=0,
                trained_label_lst=[],
                feature_count=0,
            )
        )
        self.assertIn("none", table)

    def test_lists_trained_labels(self):
        table = _format_status_table(
            DatabaseStatus(
                connection_str="sqlite:///email.db",
                db_user_id=1,
                message_count=5,
                active_message_count=4,
                deleted_message_count=1,
                trained_label_lst=["Inbox", "Spam"],
                feature_count=7,
            )
        )
        self.assertIn("Inbox, Spam", table)
        self.assertIn("2 (", table)


class IdentificationArgumentTest(TestCase):
    def test_invalid_identification_returns_config_error(self):
        exit_code = command_line_parser(["status", "-i", "not-a-number"])
        self.assertEqual(exit_code, _EXIT_CONFIG_ERROR)

    @patch("mailsort.__main__.Imap")
    def test_global_flags_before_subcommand_are_preserved(self, imap_cls):
        """
        Regression test: argparse subparsers replace the whole namespace with their own
        parsed values, so a subcommand copy of --host/--username/... with an unconditional
        default would silently wipe out the same flag given before the subcommand name -
        see the suppress_defaults handling in mailsort.__main__._add_connection_arguments.
        """
        imap_instance = imap_cls.return_value

        exit_code = command_line_parser(
            [
                "--host",
                "localhost",
                "--username",
                "user",
                "--password",
                "secret",
                "sync",
            ]
        )

        self.assertEqual(exit_code, _EXIT_OK)
        imap_cls.assert_called_once_with(
            host="localhost",
            port=993,
            username="user",
            password="secret",
            connection_str="sqlite:///email.db",
            db_user_id=1,
            use_ssl=True,
            email_download_format="metadata",
        )
        imap_instance.update_database.assert_called_once_with(quick=False, label_lst=None)


if __name__ == "__main__":
    import unittest

    unittest.main()
