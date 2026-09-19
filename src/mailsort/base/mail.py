from abc import ABC, abstractmethod

import pandas
from tqdm import tqdm

from mailsort.ml import (
    encode_df_for_machine_learning,
    fit_machine_learning_models,
    score_messages_with_machine_learning_models,
)
from mailsort.results import Prediction, SortResult, SyncResult, TrainResult


class AbstractMailBox(ABC):
    def __init__(
        self,
        mail_service,
        database_email=None,
        database_ml=None,
        user_id="me",
        db_user_id=1,
        email_download_format="metadata",
    ):
        """
        Shared fetch-store-train-predict-move loop for a mailbox backend, independent of
        whether the backend is the Gmail API or a plain IMAP connection.

        Args:
            mail_service: backend-specific connection object (Gmail API service resource,
                imaplib connection, ...)
            database_email (mailsort.base.database.DatabaseInterface): SQLalchemy interface for email database
            database_ml (mailsort.ml.database.DatabaseInterface): SQLalchemy interface for machine learning database
            user_id (str): backend-specific user identifier
            db_user_id (int): Default 1 - set a user id when sharing a database with multiple users
            email_download_format (str): backend-specific download format hint
        """
        self._service = mail_service
        self._db_email = database_email
        self._db_ml = database_ml
        self._db_user_id = db_user_id
        self._userid = user_id
        self._email_download_format = email_download_format
        self._label_dict = self._get_label_translate_dict()
        self._label_dict_inverse = {v: k for k, v in self._label_dict.items()}

    def close(self) -> None:  # noqa: B027
        """
        Release any resources held by this mailbox backend.

        The default implementation is intentionally a no-op, not abstract; override it in
        backends that hold an open connection (see mailsort.imap.mail.ImapMailBase.close() for
        the IMAP backend's override). Promoting this - and the context manager support below -
        to AbstractMailBox lets any backend, existing or future, be used as `with mailbox: ...`
        or wrapped in mailsort.MailSorter without backend-specific handling.
        """

    def __enter__(self) -> "AbstractMailBox":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    @property
    def labels(self):
        return list(self._label_dict.keys())

    def download_emails_for_label(self, label):
        """
        Download emails for a specific label

        Args:
            label (str): label to download emails for

        Returns:
            pandas.DataFrame: Email content for the downloaded emails
        """
        return self._download_messages_to_dataframe(
            message_id_lst=self._search_email_on_server(
                label_lst=[label], only_message_ids=True
            )
        )

    def filter_messages_from_server(
        self,
        label,
        recommendation_ratio=0.9,
        label_prefix: str = "labels_",
    ):
        """
        Filter new emails based on machine learning model recommendations.

        This moves messages on the server, but does no inference of its own: it scores messages
        through get_label_recommendations() - the exact same inference step
        'mailsort predict'/MailSorter.predict() use - and then moves only the accepted
        Predictions, so inference and mailbox mutation stay clearly separated in code, not just
        by convention. To preview what this would do, without moving, deleting or otherwise
        modifying anything, call get_label_recommendations() directly.

        Args:
            label (str): Email label to filter for
            recommendation_ratio (float): Only accept recommendation above this ratio (0<r<1)

        Returns:
            mailsort.results.SortResult: which messages were actually moved, and to where
        """
        prediction_lst = self.get_label_recommendations(
            label=label,
            recommendation_ratio=recommendation_ratio,
            label_prefix=label_prefix,
        )
        accepted_lst = [
            prediction for prediction in prediction_lst if prediction.accepted
        ]
        moved_lst = self._move_emails(
            prediction_lst=accepted_lst, label_to_ignore=label
        )
        return SortResult(moved_lst=moved_lst)

    def get_label_recommendations(
        self,
        label,
        recommendation_ratio=0.9,
        label_prefix: str = "labels_",
    ):
        """
        Score the messages currently in `label` against the trained machine learning models,
        without moving, deleting, archiving or otherwise modifying anything on the mail server -
        a read-only dry run of filter_messages_from_server(), and the exact same inference step
        that method's moves are based on.

        This reuses the exact same download and feature-encoding steps as
        filter_messages_from_server(), and the exact same underlying per-label model scores (see
        mailsort.ml.model.score_messages_with_machine_learning_models), so `accepted` below always
        agrees with whether filter_messages_from_server() would move that message for real, given
        the same recommendation_ratio - only the move itself is left out.

        Args:
            label (str): Email label/folder to fetch and score messages from
            recommendation_ratio (float): Cutoff ratio (0<r<1) a score must clear for `accepted`
                to be True - the same cutoff filter_messages_from_server() uses to decide whether
                to actually move a message
            label_prefix (str): prefix used to recognise label columns during feature encoding

        Returns:
            list[mailsort.results.Prediction]: one Prediction per message currently in `label`
        """
        df_partial = self.download_emails_for_label(label=label)
        if len(df_partial) == 0:
            return []
        model_reload_dict, feature_reload_lst = self._db_ml.load_models()
        if len(model_reload_dict) == 0:
            # No machine learning model has been trained yet (fit_machine_learning_model_to_database()
            # was never run) - nothing to score against, so every message abstains rather than
            # encoding features for a model that does not exist.
            return [
                Prediction(
                    message_id=message_id,
                    source_folder=label,
                    recommended_folder=None,
                    score=0.0,
                    threshold=recommendation_ratio,
                    accepted=False,
                    subject=subject,
                )
                for message_id, subject in zip(
                    df_partial["id"], df_partial["subject"], strict=False
                )
            ]
        df_partial_features = encode_df_for_machine_learning(
            df=df_partial,
            feature_lst=feature_reload_lst,
            label_lst=list(model_reload_dict.keys()),
            return_labels=False,
            label_prefix=label_prefix,
        )
        df_partial_features = df_partial_features.reindex(
            sorted(df_partial_features.columns), axis=1
        )
        score_lst = score_messages_with_machine_learning_models(
            df_features=df_partial_features,
            model_dict=model_reload_dict,
            recommendation_ratio=recommendation_ratio,
        )
        subject_by_id = dict(zip(df_partial["id"], df_partial["subject"], strict=False))
        return [
            Prediction(
                message_id=entry["email_id"],
                source_folder=label,
                recommended_folder=entry["recommended_label"],
                score=entry["score"],
                threshold=recommendation_ratio,
                accepted=entry["threshold_reached"],
                subject=subject_by_id.get(entry["email_id"]),
            )
            for entry in score_lst
        ]

    def fit_machine_learning_model_to_database(
        self,
        n_estimators=100,
        max_features=400,
        random_state=42,
        bootstrap=True,
        include_deleted=False,
        max_workers=None,
    ):
        """
        Fit machine learning models to emails stored in database and afterwards store machine learning models in
        database.

        Args:
            n_estimators (int): Number of estimators
            max_features (int): Number of features
            random_state (int): Random state
            bootstrap (boolean): Whether bootstrap samples are used when building trees. If False, the whole dataset is
                                 used to build each tree. (default: true)
            include_deleted (bool): Flag to include deleted emails - default False
            max_workers (int): maximum number of workers for the machine learning models

        Returns:
            mailsort.results.TrainResult: the folders/labels a model was trained for
        """
        df_all = self.get_all_emails_in_database(include_deleted=include_deleted)
        df_all_features, df_all_labels = encode_df_for_machine_learning(
            df=df_all, feature_lst=[], label_lst=[], return_labels=True
        )
        df_all_features = df_all_features.loc[
            :, ~df_all_features.columns.duplicated()
        ].copy()
        df_all_features = df_all_features.reindex(
            sorted(df_all_features.columns), axis=1
        )
        model_dict = fit_machine_learning_models(
            df_all_features=df_all_features,
            df_all_labels=df_all_labels,
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_state,
            bootstrap=bootstrap,
            max_workers=max_workers,
        )
        self._db_ml.store_models(
            model_dict=model_dict,
            feature_lst=df_all_features.columns.values.tolist(),
            user_id=self._db_user_id,
            commit=True,
        )
        return TrainResult(trained_label_lst=sorted(model_dict.keys()))

    def get_all_emails_in_database(self, include_deleted=False):
        """
        Get all emails stored in the local database

        Args:
            include_deleted (bool): Flag to include deleted emails - default False

        Returns:
            pandas.DataFrame: With all emails and the corresponding information
        """
        return self._db_email.get_all_emails(
            include_deleted=include_deleted, user_id=self._db_user_id
        )

    def update_database(self, quick=False, label_lst=None, email_format=None):
        """
        Update local email database

        Args:
            quick (boolean): Only add new emails, do not update existing labels - by default: False
            label_lst (list): list of labels to be searched
            email_format (str/None): Email format to download

        Returns:
            mailsort.results.SyncResult: counts of messages newly stored, relabeled and marked
                deleted by this call - relabeling and deletion counts are always 0 when
                quick=True, since that mode skips both steps
        """
        if label_lst is None:
            label_lst = []
        if self._db_email is None:
            return SyncResult(
                new_message_count=0, updated_message_count=0, deleted_message_count=0
            )
        message_id_lst = self._search_email_on_server(
            label_lst=label_lst, only_message_ids=True
        )
        (
            new_messages_lst,
            message_label_updates_lst,
            deleted_messages_lst,
        ) = self._db_email.get_labels_to_update(
            message_id_lst=message_id_lst, user_id=self._db_user_id
        )
        updated_message_count = 0
        deleted_message_count = 0
        if not quick:
            self._db_email.mark_emails_as_deleted(
                message_id_lst=deleted_messages_lst, user_id=self._db_user_id
            )
            self._db_email.update_labels(
                message_id_lst=message_label_updates_lst,
                message_meta_lst=self._get_labels_for_emails(
                    message_id_lst=message_label_updates_lst
                ),
                user_id=self._db_user_id,
            )
            updated_message_count = len(message_label_updates_lst)
            deleted_message_count = len(deleted_messages_lst)
        self._store_emails_in_database(
            message_id_lst=new_messages_lst, email_format=email_format
        )
        return SyncResult(
            new_message_count=len(new_messages_lst),
            updated_message_count=updated_message_count,
            deleted_message_count=deleted_message_count,
        )

    def _download_messages_to_dataframe(self, message_id_lst, email_format=None):
        """
        Download a list of messages based on their email IDs and store the content in a pandas.DataFrame.

        Args:
            message_id_lst (list): list of emails IDs
            email_format (str): Email format to download - default: "full"

        Returns:
            pandas.DataFrame: pandas.DataFrame which contains the rendered emails
        """
        return pandas.DataFrame(
            [
                message
                for message in [
                    self._parse_message(
                        message=self._get_message_detail(
                            message_id=message_id,
                            email_format=email_format,
                            metadata_headers=[],
                        )
                    )
                    for message_id in tqdm(
                        iterable=message_id_lst, desc="Download messages to DataFrame"
                    )
                ]
                if message is not None
            ]
        )

    def _get_labels_for_emails(self, message_id_lst):
        """
        Get labels for a list of emails

        Args:
            message_id_lst (list): list of emails IDs

        Returns:
            list: Nested list of email labels for each email
        """
        return [
            self._get_labels_for_email(message_id=message_id)
            for message_id in tqdm(
                iterable=message_id_lst, desc="Get labels for emails"
            )
        ]

    def _move_emails(self, prediction_lst, label_to_ignore):
        """
        Move messages to their recommended folder - the only place inference results (Predictions)
        are turned into mailbox mutation, so this consumes Predictions rather than scoring
        anything itself.

        Args:
            prediction_lst (list[mailsort.results.Prediction]): predictions to act on - callers
                are expected to have already filtered this to accepted predictions, see
                filter_messages_from_server()
            label_to_ignore (str): folder a message must not be "moved" to because it is already
                there

        Returns:
            list[tuple[str, str]]: (message_id, label) pairs actually moved
        """
        label_existing = self._label_dict[label_to_ignore]
        moved_lst = []
        for prediction in tqdm(iterable=prediction_lst, desc="Move emails"):
            label_add = prediction.recommended_folder
            if label_add is not None and label_add != label_existing:
                self._modify_message_labels(
                    message_id=prediction.message_id,
                    label_id_remove_lst=[label_existing],
                    label_id_add_lst=[label_add],
                )
                moved_lst.append((prediction.message_id, label_add))
        return moved_lst

    def _store_emails_in_database(self, message_id_lst, email_format=None):
        df = self._download_messages_to_dataframe(
            message_id_lst=message_id_lst, email_format=email_format
        )
        if len(df) > 0:
            self._db_email.store_dataframe(df=df, user_id=self._db_user_id)

    @abstractmethod
    def _search_email_on_server(
        self, query_string="", label_lst=None, only_message_ids=False
    ):
        """
        Search emails either by a specific query or optionally limit your search to a list of labels

        Args:
            query_string (str): query string to search for
            label_lst (list): list of labels to be searched
            only_message_ids (bool): return only the email IDs not the thread IDs - default: false

        Returns:
            list: list of message ids (or backend-specific list items) matching the search
        """

    @abstractmethod
    def _get_message_detail(self, message_id, email_format=None, metadata_headers=None):
        """
        Get the raw, backend-specific representation of a single email message.

        Args:
            message_id (str): id used by this backend to uniquely identify the email
            email_format (str/None): backend-specific format hint
            metadata_headers (list): backend-specific list of metadata headers

        Returns:
            The backend-specific raw message representation, passed on to `_parse_message`.
        """

    @abstractmethod
    def _get_label_translate_dict(self):
        """
        Returns:
            dict: mapping of label/folder display name to the backend-specific label/folder id
        """

    @abstractmethod
    def _modify_message_labels(
        self, message_id, label_id_remove_lst=None, label_id_add_lst=None
    ):
        """
        Apply a label/folder change to a single email message.
        """

    @abstractmethod
    def _get_labels_for_email(self, message_id):
        """
        Args:
            message_id (str): id used by this backend to uniquely identify the email

        Returns:
            list: list of labels/folders currently assigned to the email
        """

    @abstractmethod
    def _parse_message(self, message):
        """
        Args:
            message: the backend-specific raw message representation returned by `_get_message_detail`

        Returns:
            dict/None: the common mailsort email dict (see `mailsort.base.message.AbstractMessage.to_dict`),
                       or None if the message could not be parsed
        """
