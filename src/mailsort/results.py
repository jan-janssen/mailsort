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
    A single machine learning recommendation for one message.

    See AbstractMailBox.get_label_recommendations() / mailsort.MailSorter.predict(). Producing a
    Prediction never moves, deletes or otherwise modifies anything on the mail server - see
    SortResult for the outcome of actually acting on recommendations like this one.

    Attributes:
        message_id (str): backend-specific id that uniquely identifies the message
        subject (str/None): the message subject, if available
        recommended_label (str/None): the label the model scores highest for this message, or
            None if no machine learning model has been trained yet
        score (float): the model's score for `recommended_label`
        threshold_reached (bool): whether `score` clears the recommendation_ratio used to
            produce this prediction, i.e. whether mailsort.MailSorter.sort() would move this
            message for real, given the same recommendation_ratio
    """

    message_id: str
    subject: str | None
    recommended_label: str | None
    score: float
    threshold_reached: bool


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
