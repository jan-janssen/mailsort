from mailsort.local import Imap
from mailsort.results import Prediction, SortResult, SyncResult, TrainResult
from mailsort.sorter import MailSorter

from . import _version

__version__: str = _version.__version__
__all__ = [
    "Imap",
    "MailSorter",
    "Prediction",
    "SortResult",
    "SyncResult",
    "TrainResult",
]
