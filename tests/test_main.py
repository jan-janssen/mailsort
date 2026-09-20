from unittest import TestCase

from mailsort.__main__ import _format_recommendations_table
from mailsort.results import Prediction, ScoreType


class FormatRecommendationsTableTest(TestCase):
    def test_empty_recommendations(self):
        self.assertEqual(
            _format_recommendations_table([]), "No messages found in this folder."
        )

    def test_includes_every_message_and_field(self):
        table = _format_recommendations_table(
            [
                Prediction(
                    message_id="INBOX\x1f1",
                    source_folder="INBOX",
                    recommended_folder="Sorted",
                    score=1.0,
                    threshold=0.9,
                    accepted=True,
                    score_type=ScoreType.CALIBRATED,
                    subject="Hello",
                ),
                Prediction(
                    message_id="INBOX\x1f2",
                    source_folder="INBOX",
                    recommended_folder=None,
                    score=0.0,
                    threshold=0.9,
                    accepted=False,
                    score_type=ScoreType.RAW,
                    subject=None,
                ),
            ]
        )

        self.assertIn("INBOX\x1f1", table)
        self.assertIn("Hello", table)
        self.assertIn("Sorted", table)
        self.assertIn("True", table)
        self.assertIn("calibrated", table)
        self.assertIn("INBOX\x1f2", table)
        self.assertIn("False", table)
        self.assertIn("raw", table)
        # a missing subject/recommendation must not crash formatting or print "None"
        self.assertNotIn("None", table)


if __name__ == "__main__":
    import unittest

    unittest.main()
