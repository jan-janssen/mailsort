import pickle
from datetime import datetime
from unittest import TestCase

import pandas
from sqlalchemy import String, Text, create_engine, make_url
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateTable

from mailsort.base.database import Base as EmailBase
from mailsort.base.database import _with_mysql_charset, get_email_database
from mailsort.ml.database import Base as MachineLearningBase
from mailsort.ml.database import get_machine_learning_database


def _mysql_column_types(base):
    dialect = mysql.dialect()
    return {
        f"{table.name}.{column.name}": column.type.compile(dialect=dialect)
        for table in base.metadata.sorted_tables
        for column in table.columns
    }


class MySQLSchemaTest(TestCase):
    def test_all_string_columns_have_length_or_are_text(self):
        for base in [EmailBase, MachineLearningBase]:
            for table in base.metadata.sorted_tables:
                for column in table.columns:
                    if isinstance(column.type, String) and not isinstance(
                        column.type, Text
                    ):
                        self.assertIsNotNone(
                            column.type.length, f"{table.name}.{column.name}"
                        )

    def test_email_column_types(self):
        column_types = _mysql_column_types(EmailBase)
        for column in [
            "email_content.email_id",
            "email_threads.email_id",
            "email_threads.thread_id",
            "email_labels.email_id",
            "email_labels.label_id",
            "email_to.email_id",
            "email_cc.email_id",
            "email_from.email_id",
            "email_from.email_from",
        ]:
            self.assertEqual(column_types[column], "VARCHAR(255)", column)
        for column in [
            "email_content.email_subject",
            "email_to.email_to",
            "email_cc.email_cc",
        ]:
            self.assertEqual(column_types[column], "TEXT", column)
        self.assertEqual(column_types["email_content.email_content"], "LONGTEXT")

    def test_machine_learning_column_types(self):
        column_types = _mysql_column_types(MachineLearningBase)
        self.assertEqual(column_types["ml_labels.label_id"], "VARCHAR(255)")
        self.assertEqual(column_types["ml_labels.random_forest"], "LONGBLOB")
        self.assertEqual(column_types["ml_features.feature"], "TEXT")

    def test_create_table_compiles_for_mysql(self):
        dialect = mysql.dialect()
        for base in [EmailBase, MachineLearningBase]:
            for table in base.metadata.sorted_tables:
                ddl = str(CreateTable(table).compile(dialect=dialect))
                self.assertNotIn("VARCHAR,", ddl)
                self.assertNotIn("VARCHAR\n", ddl)

    def test_foreign_key_target_is_unique(self):
        # MySQL 8.4 requires a unique key on the column referenced by a foreign key
        table = EmailBase.metadata.tables["email_content"]
        ddl = str(CreateTable(table).compile(dialect=mysql.dialect()))
        self.assertIn("UNIQUE (email_id)", ddl)

    def test_tables_use_utf8mb4(self):
        # MySQL's utf8 charset only supports up to 3 bytes per character, so
        # 4 byte characters like emojis require utf8mb4
        dialect = mysql.dialect()
        for base in [EmailBase, MachineLearningBase]:
            for table in base.metadata.sorted_tables:
                ddl = str(CreateTable(table).compile(dialect=dialect))
                self.assertIn("CHARSET=utf8mb4", ddl, table.name)
                self.assertIn("COLLATE utf8mb4_bin", ddl, table.name)


class MySQLConnectionCharsetTest(TestCase):
    def test_mysql_connection_uses_utf8mb4(self):
        for connection_str in [
            "mysql://user:password@localhost/mailsort",
            "mysql+mysqldb://user:password@localhost/mailsort",
            "mysql+pymysql://user:password@localhost/mailsort",
            "mariadb+mysqldb://user:password@localhost/mailsort",
        ]:
            url = _with_mysql_charset(connection_str)
            self.assertEqual(url.query["charset"], "utf8mb4", connection_str)

    def test_explicit_charset_is_kept(self):
        url = _with_mysql_charset(
            "mysql+mysqldb://user:password@localhost/mailsort?charset=latin1"
        )
        self.assertEqual(url.query["charset"], "latin1")

    def test_other_databases_are_unchanged(self):
        for connection_str in ["sqlite:///:memory:", "sqlite:///mailsort.db"]:
            self.assertEqual(
                _with_mysql_charset(connection_str), make_url(connection_str)
            )


class LongValuesRoundTripTest(TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        self.session = sessionmaker(bind=self.engine)()

    def tearDown(self) -> None:
        self.session.close()

    def test_unicode_email_fields(self):
        df = pandas.DataFrame(
            [
                {
                    "content": "Hello 🌍 — café",
                    "date": datetime(2022, 2, 11, 18, 8, 46),
                    "from": "sender@example.com",
                    "id": "unicode-email",
                    "cc": ["café@example.com"],
                    "labels": ["重要📬"],
                    "subject": "こんにちは 🌟",
                    "threads": "thread-😀",
                    "to": ["recipient📬@example.com"],
                }
            ]
        )
        database = get_email_database(engine=self.engine, session=self.session)
        database.store_dataframe(df=df)

        df_stored = database.get_all_emails()
        self.assertEqual(df_stored["subject"].values[0], "こんにちは 🌟")
        self.assertEqual(df_stored["content"].values[0], "Hello 🌍 — café")
        self.assertEqual(df_stored["labels"].values[0], ["重要📬"])
        self.assertEqual(df_stored["threads"].values[0], "thread-😀")
        self.assertEqual(df_stored["to"].values[0], ["recipient📬@example.com"])

    def test_long_email_fields(self):
        to_lst = [",".join(f"user{i}@example.com" for i in range(300))]
        subject = "s" * 1000
        content = "c" * 100000
        df = pandas.DataFrame(
            [
                {
                    "content": content,
                    "date": datetime(2022, 2, 11, 18, 8, 46),
                    "from": "sender@server.net",
                    "id": "myid123",
                    "cc": to_lst,
                    "labels": ["important"],
                    "subject": subject,
                    "threads": "abc123",
                    "to": to_lst,
                }
            ]
        )
        database = get_email_database(engine=self.engine, session=self.session)
        database.store_dataframe(df=df)
        df_stored = database.get_all_emails()
        self.assertEqual(df_stored["subject"].values[0], subject)
        self.assertEqual(df_stored["content"].values[0], content)
        self.assertEqual(df_stored["to"].values[0], to_lst)
        self.assertEqual(df_stored["cc"].values[0], to_lst)

    def test_long_features_and_binary_models(self):
        feature = "to_" + ",".join(f"user{i}@example.com" for i in range(300))
        model = {"weights": list(range(1000))}
        database = get_machine_learning_database(
            engine=self.engine, session=self.session
        )
        database.store_models(
            model_dict={"label": model}, feature_lst=[feature, "email_id"]
        )
        model_dict, feature_lst = database.load_models()
        self.assertEqual(model_dict, {"label": model})
        self.assertEqual(feature_lst, [feature])
        self.assertIsInstance(pickle.dumps(model), bytes)
