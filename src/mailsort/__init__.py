from mailsort.local import Imap
from mailsort.results import Prediction, ScoreType, SortResult, SyncResult, TrainResult
from mailsort.sorter import MailSorter

from . import _version

__version__: str = _version.__version__
__all__ = [
    "Imap",
    "MailSorter",
    "Prediction",
    "ScoreType",
    "SortResult",
    "SyncResult",
    "TrainResult",
]
