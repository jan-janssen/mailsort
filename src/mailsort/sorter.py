"""
High-level, task-oriented facade over mailsort.base.mail.AbstractMailBox.

mailsort.Imap (and any future mailbox backend built on AbstractMailBox) already exposes the full
fetch-store-train-predict-move loop through implementation-oriented methods such as
update_database(), fit_machine_learning_model_to_database() and filter_messages_from_server().
MailSorter wraps any such backend and renames that same loop to the vocabulary an application
developer thinks in - sync/train/predict/sort - without re-implementing any of it.
"""

from __future__ import annotations

from types import TracebackType

from mailsort.base.mail import AbstractMailBox
from mailsort.results import Prediction, SortResult, SyncResult, TrainResult


class MailSorter:
    """
    Task-oriented facade over an AbstractMailBox backend (e.g. mailsort.Imap).

    MailSorter itself is backend-agnostic - it only calls the public AbstractMailBox interface,
    so it works with any current or future mailbox backend without modification:

        >>> from mailsort import Imap, MailSorter
        >>> with MailSorter(Imap(host="imap.example.com", port=993, username="user@example.com",
        ...                      password="...", connection_str="sqlite:///email.db")) as sorter:
        ...     sorter.sync()
        ...     sorter.train()
        ...     predictions = sorter.predict("MailSortInbox")
        ...     sorter.sort("MailSortInbox")

    predict() is strictly read-only - it never changes the mailbox. sort() is the only method
    that moves messages; see its docstring for how it relates to predict().
    """

    def __init__(self, mailbox: AbstractMailBox) -> None:
        """
        Args:
            mailbox: an already-constructed AbstractMailBox backend, e.g. mailsort.Imap(...).
                MailSorter takes ownership of it for the purpose of close()/the context manager
                below, but does not construct it itself, so any backend-specific connection
                arguments (host, credentials, ...) are handled by that backend, not by MailSorter.
        """
        self._mailbox = mailbox

    @property
    def mailbox(self) -> AbstractMailBox:
        """The wrapped AbstractMailBox backend, for access to lower-level operations."""
        return self._mailbox

    def close(self) -> None:
        """Release the resources held by the wrapped mailbox backend."""
        self._mailbox.close()

    def __enter__(self) -> MailSorter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.close()
        return False

    def sync(
        self,
        quick: bool = False,
        label_lst: list[str] | None = None,
        email_format: str | None = None,
    ) -> SyncResult:
        """
        Update the local database from the mail server - the "fetch/store" steps of the loop.

        Args:
            quick: only add new messages, skip re-checking the labels of already-known ones -
                default: False
            label_lst: restrict the sync to these folders - default: every folder
            email_format: backend-specific download format hint

        Returns:
            SyncResult: how many messages were newly stored, relabeled and marked deleted
        """
        return self._mailbox.update_database(
            quick=quick, label_lst=label_lst, email_format=email_format
        )

    def train(
        self,
        n_estimators: int = 100,
        max_features: int = 400,
        random_state: int = 42,
        bootstrap: bool = True,
        include_deleted: bool = False,
        max_workers: int | None = None,
    ) -> TrainResult:
        """
        (Re-)train one machine learning model per folder on the messages in the local database -
        the "train" step of the loop.

        Args:
            n_estimators: number of estimators per model
            max_features: maximum number of features per model
            random_state: random state for initialization
            bootstrap: whether bootstrap samples are used when building trees
            include_deleted: include messages marked as deleted - default: False
            max_workers: maximum number of worker processes to train models in parallel

        Returns:
            TrainResult: the folders a model was trained for
        """
        return self._mailbox.fit_machine_learning_model_to_database(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_state,
            bootstrap=bootstrap,
            include_deleted=include_deleted,
            max_workers=max_workers,
        )

    def predict(
        self,
        folder: str,
        recommendation_ratio: float = 0.9,
        label_prefix: str = "labels_",
    ) -> list[Prediction]:
        """
        Report machine learning recommendations for the messages currently in `folder` - the
        "predict" step of the loop, read-only.

        This never moves, deletes, archives or otherwise modifies anything on the mail server -
        it is safe to call at any time, including before you trust the model with your mailbox.
        See sort() to act on these recommendations instead of only reporting them.

        Args:
            folder: mail folder to fetch and score messages from
            recommendation_ratio: certainty a score must clear for `accepted` to be True on the
                matching Prediction - the same cutoff sort() uses to decide whether to actually
                move a message (0<r<1)
            label_prefix: prefix used to recognise label columns during feature encoding

        Returns:
            list[Prediction]: one entry per message currently in `folder`
        """
        return self._mailbox.get_label_recommendations(
            label=folder,
            recommendation_ratio=recommendation_ratio,
            label_prefix=label_prefix,
        )

    def sort(
        self,
        folder: str,
        recommendation_ratio: float = 0.9,
        label_prefix: str = "labels_",
    ) -> SortResult:
        """
        Classify the messages currently in `folder` and move the ones whose score clears
        `recommendation_ratio` to the recommended folder - the "move" step of the loop.

        This is the only MailSorter method that changes the mailbox. It scores messages exactly
        as predict() does with the same arguments, so predict() first is the way to preview what
        this call would do without moving anything.

        Args:
            folder: mail folder to fetch, score and sort messages from
            recommendation_ratio: certainty a score must clear to actually move a message (0<r<1)
            label_prefix: prefix used to recognise label columns during feature encoding

        Returns:
            SortResult: which messages were actually moved, and to where
        """
        return self._mailbox.filter_messages_from_server(
            label=folder,
            recommendation_ratio=recommendation_ratio,
            label_prefix=label_prefix,
        )
