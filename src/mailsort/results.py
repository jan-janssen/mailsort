"""
Structured result types returned by mailsort.base.mail.AbstractMailBox and the mailsort.MailSorter
facade built on top of it.

These types are backend-agnostic - any AbstractMailBox subclass (mailsort.Imap, or a future
Gmail/other backend) reports its sync/train/predict/sort results through the same types, so
callers do not need backend-specific handling.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SyncResult:
    """
    Result of syncing the local database with the mail server.

    See AbstractMailBox.update_database() / mailsort.MailSorter.sync().

    Attributes:
        new_message_count (int): messages downloaded and stored for the first time
        updated_message_count (int): already-known messages whose labels changed - always 0
            when the sync was run with quick=True, since that mode skips relabeling
        deleted_message_count (int): already-known messages marked deleted because they no
            longer exist on the server - always 0 when the sync was run with quick=True
    """

    new_message_count: int
    updated_message_count: int
    deleted_message_count: int


@dataclass(frozen=True)
class TrainResult:
    """
    Result of (re-)training the machine learning models.

    See AbstractMailBox.fit_machine_learning_model_to_database() / mailsort.MailSorter.train().

    Attributes:
        trained_label_lst (list[str]): folders/labels a model was trained for, sorted
    """

    trained_label_lst: list[str]

    @property
    def model_count(self) -> int:
        """Number of per-folder models trained."""
        return len(self.trained_label_lst)


@dataclass(frozen=True)
class Prediction:
    """
    A single machine learning classification result for one message - a first-class,
    side-effect-free representation of inference, deliberately kept separate from mailbox
    mutation.

    See AbstractMailBox.get_label_recommendations() / mailsort.MailSorter.predict(), which
    produce these; producing a Prediction never moves, deletes or otherwise modifies anything on
    the mail server. Moving messages (AbstractMailBox.filter_messages_from_server() /
    mailsort.MailSorter.sort()) consumes the `accepted` Predictions from that same inference step
    rather than scoring messages again, so a Prediction is authoritative for what sort() would do
    given the same recommendation_ratio - see SortResult for the outcome of actually acting on
    predictions like this one.

    Every field is a plain, JSON-serializable value (str/float/bool, never a scikit-learn object),
    so a Prediction can be logged, displayed or sent over the wire as-is - by the CLI, a caller
    such as gmailsorter, a future web interface, or an audit trail.

    Attributes:
        message_id (str): backend-specific id that uniquely identifies the message
        source_folder (str): folder the message was fetched and scored from
        recommended_folder (str/None): the folder the model scores highest for this message, or
            None if no machine learning model has been trained yet
        score (float): the model's score for `recommended_folder` (0.0 when `recommended_folder`
            is None)
        threshold (float): the recommendation_ratio this prediction was scored against - carried
            alongside `score` so a Prediction is self-contained: a caller does not need to
            remember which recommendation_ratio produced it to know why `accepted` is what it is
        accepted (bool): whether `score` strictly clears `threshold`, i.e. whether
            filter_messages_from_server()/MailSorter.sort() would move this message for real,
            given the same recommendation_ratio - an abstained prediction (accepted=False) is
            never acted on
        subject (str/None): the message subject, if available - display metadata for humans
            (CLI/logging), not itself part of the classification
    """

    message_id: str
    source_folder: str
    recommended_folder: str | None
    score: float
    threshold: float
    accepted: bool
    subject: str | None = None


@dataclass(frozen=True)
class SortResult:
    """
    Result of moving messages based on machine learning recommendations.

    See AbstractMailBox.filter_messages_from_server() / mailsort.MailSorter.sort().

    Attributes:
        moved_lst (list[tuple[str, str]]): (message_id, label) pairs actually moved - a subset
            of the predictions that reached the recommendation threshold, since a message already
            sitting in its recommended label is left alone
    """

    moved_lst: list[tuple[str, str]]

    @property
    def moved_count(self) -> int:
        """Number of messages actually moved."""
        return len(self.moved_lst)
