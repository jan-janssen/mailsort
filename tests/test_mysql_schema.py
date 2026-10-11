import pickle
from datetime import datetime
from unittest import TestCase

import pandas
from sqlalchemy import String, Text, create_engine
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateIndex, CreateTable

from mailsort.base.database import Base as EmailBase
from mailsort.base.database import get_email_database
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

    def test_foreign_key_target_is_indexed(self):
        # MySQL requires an index on the column referenced by a foreign key
        table = EmailBase.metadata.tables["email_content"]
        ddl = [
            str(CreateIndex(index).compile(dialect=mysql.dialect()))
            for index in table.indexes
        ]
        self.assertTrue(any("(email_id)" in statement for statement in ddl))


class LongValuesRoundTripTest(TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        self.session = sessionmaker(bind=self.engine)()

    def tearDown(self) -> None:
        self.session.close()

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
