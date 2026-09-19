from unittest import TestCase
from unittest.mock import patch

from mailsort.__main__ import _format_recommendations_table, command_line_parser


class ImapCliTest(TestCase):
    @patch("mailsort.__main__.Imap")
    def test_update_wires_imap_and_triggers_update(self, imap_cls):
        imap_instance = imap_cls.return_value
        with patch(
            "sys.argv",
            [
                "mailsort",
                "--host",
                "localhost",
                "--port",
                "993",
                "--username",
                "user",
                "--password",
                "secret",
                "-d",
                "sqlite:///:memory:",
                "-u",
            ],
        ):
            command_line_parser()

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
        imap_instance.update_database.assert_called_once_with(quick=False)
        imap_instance.fit_machine_learning_model_to_database.assert_called_once_with(
            n_estimators=100,
            max_features=400,
            random_state=42,
            bootstrap=True,
            include_deleted=False,
        )

    @patch("mailsort.__main__.Imap")
    def test_label_wires_imap_and_triggers_filter(self, imap_cls):
        imap_instance = imap_cls.return_value
        with patch(
            "sys.argv",
            [
                "mailsort",
                "--host",
                "localhost",
                "--username",
                "user",
                "--password",
                "secret",
                "-d",
                "sqlite:///:memory:",
                "-l",
                "MailSortInbox",
            ],
        ):
            command_line_parser()

        imap_instance.filter_messages_from_server.assert_called_once_with(
            label="MailSortInbox",
            recommendation_ratio=0.9,
            label_prefix="labels_",
        )

    @patch("mailsort.__main__.Imap")
    def test_dry_run_prints_recommendations_and_never_filters(self, imap_cls):
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
        with (
            patch(
                "sys.argv",
                [
                    "mailsort",
                    "--host",
                    "localhost",
                    "--username",
                    "user",
                    "--password",
                    "secret",
                    "-d",
                    "sqlite:///:memory:",
                    "-l",
                    "MailSortInbox",
                    "--dry-run",
                ],
            ),
            patch("builtins.print") as print_mock,
        ):
            command_line_parser()

        imap_instance.get_label_recommendations.assert_called_once_with(
            label="MailSortInbox",
            recommendation_ratio=0.9,
            label_prefix="labels_",
        )
        imap_instance.filter_messages_from_server.assert_not_called()
        printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list)
        self.assertIn("Hello", printed)
        self.assertIn("Sorted", printed)

    @patch("mailsort.__main__.Imap")
    def test_no_update_or_label_prints_help(self, imap_cls):
        imap_instance = imap_cls.return_value
        with patch(
            "sys.argv",
            [
                "mailsort",
                "--host",
                "localhost",
                "--username",
                "user",
                "--password",
                "secret",
                "-d",
                "sqlite:///:memory:",
            ],
        ):
            command_line_parser()

        imap_instance.update_database.assert_not_called()
        imap_instance.filter_messages_from_server.assert_not_called()

    @patch("mailsort.__main__.Imap")
    def test_missing_password_skips_wiring(self, imap_cls):
        with patch(
            "sys.argv",
            ["mailsort", "--host", "localhost", "--username", "user"],
        ):
            command_line_parser()

        imap_cls.assert_not_called()

    @patch("mailsort.__main__.Imap")
    def test_missing_host_skips_wiring(self, imap_cls):
        with patch("sys.argv", ["mailsort", "--username", "user"]):
            command_line_parser()

        imap_cls.assert_not_called()


class FormatRecommendationsTableTest(TestCase):
    def test_empty_recommendations(self):
        self.assertEqual(
            _format_recommendations_table([]), "No messages found in this folder."
        )

    def test_includes_every_message_and_field(self):
        table = _format_recommendations_table(
            [
                {
                    "message_id": "INBOX\x1f1",
                    "subject": "Hello",
                    "recommended_label": "Sorted",
                    "score": 1.0,
                    "threshold_reached": True,
                },
                {
                    "message_id": "INBOX\x1f2",
                    "subject": None,
                    "recommended_label": None,
                    "score": 0.0,
                    "threshold_reached": False,
                },
            ]
        )

        self.assertIn("INBOX\x1f1", table)
        self.assertIn("Hello", table)
        self.assertIn("Sorted", table)
        self.assertIn("True", table)
        self.assertIn("INBOX\x1f2", table)
        self.assertIn("False", table)
        # a missing subject/recommendation must not crash formatting or print "None"
        self.assertNotIn("None", table)


if __name__ == "__main__":
    import unittest

    unittest.main()
