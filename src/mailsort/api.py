"""
Stable import surface for packages that build on top of mailsort, such as gmailsorter.

Downstream packages should import from here rather than reaching into mailsort's
internal modules directly, so that mailsort's internals can be reorganised without
breaking those packages, as long as this module is kept up to date.
"""

from mailsort.base import get_email_database
from mailsort.base.database import DatabaseInterface, DatabaseTemplate
from mailsort.base.mail import AbstractMailBox
from mailsort.base.message import AbstractMessage, email_date_converter, strip_html_tags
from mailsort.ml import get_machine_learning_database
from mailsort.ml.database import MachineLearningDatabase
from mailsort.status import DatabaseStatus, get_database_status
from mailsort.training import train_machine_learning_models

__all__ = [
    "AbstractMailBox",
    "AbstractMessage",
    "DatabaseInterface",
    "DatabaseStatus",
    "DatabaseTemplate",
    "MachineLearningDatabase",
    "email_date_converter",
    "get_database_status",
    "get_email_database",
    "get_machine_learning_database",
    "strip_html_tags",
    "train_machine_learning_models",
]
