from unittest import TestCase

from mailsort import api
from mailsort.base import get_email_database
from mailsort.base.database import DatabaseInterface, DatabaseTemplate
from mailsort.base.mail import AbstractMailBox
from mailsort.base.message import AbstractMessage, email_date_converter, strip_html_tags
from mailsort.ml import get_machine_learning_database
from mailsort.ml.database import MachineLearningDatabase


class ApiTest(TestCase):
    """
    mailsort.api is the stable import surface used by downstream packages such as
    gmailsorter. This test guards against silently breaking that surface when
    mailsort's internal modules are reorganised.
    """

    def test_exports_match_underlying_objects(self):
        self.assertIs(api.get_email_database, get_email_database)
        self.assertIs(api.DatabaseInterface, DatabaseInterface)
        self.assertIs(api.DatabaseTemplate, DatabaseTemplate)
        self.assertIs(api.AbstractMailBox, AbstractMailBox)
        self.assertIs(api.AbstractMessage, AbstractMessage)
        self.assertIs(api.email_date_converter, email_date_converter)
        self.assertIs(api.strip_html_tags, strip_html_tags)
        self.assertIs(api.get_machine_learning_database, get_machine_learning_database)
        self.assertIs(api.MachineLearningDatabase, MachineLearningDatabase)

    def test_all_matches_exported_names(self):
        self.assertEqual(
            sorted(api.__all__),
            [
                "AbstractMailBox",
                "AbstractMessage",
                "DatabaseInterface",
                "DatabaseTemplate",
                "MachineLearningDatabase",
                "email_date_converter",
                "get_email_database",
                "get_machine_learning_database",
                "strip_html_tags",
            ],
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
