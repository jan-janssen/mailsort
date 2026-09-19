# mailsort
[![Python package](https://github.com/jan-janssen/mailsort/actions/workflows/unittest.yml/badge.svg?branch=main)](https://github.com/jan-janssen/mailsort/actions/workflows/unittest.yml)
[![codecov](https://codecov.io/github/jan-janssen/mailsort/graph/badge.svg)](https://codecov.io/github/jan-janssen/mailsort)
[![Documentation Status](https://readthedocs.org/projects/mailsort/badge/?version=latest)](https://mailsort.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Assign labels to emails on any IMAP mail server based on their similarity to other emails already
assigned to the same label.

`mailsort` connects to a mail account over plain IMAP, trains a machine learning model on the labels
(IMAP folders) you have already assigned, and uses that model to suggest or apply labels to new
messages. It has no dependency on Google APIs - for Gmail-specific features (OAuth, the Gmail label
API, the sorting daemon and web UI) see [gmailsorter](https://github.com/jan-janssen/gmailsorter),
which depends on `mailsort` for the shared IMAP and machine learning core.

To learn more about `mailsort` please have a look at the documentation below.

* [Preparation](https://mailsort.readthedocs.io/en/latest/preparation.html)
  * [Sort your emails](https://mailsort.readthedocs.io/en/latest/preparation.html#sort-your-emails)
  * [Configure your email account](https://mailsort.readthedocs.io/en/latest/preparation.html#configure-your-email-account)
* [Configuration](https://mailsort.readthedocs.io/en/latest/configuration.html)
  * [Command line interface](https://mailsort.readthedocs.io/en/latest/configuration.html#command-line-interface)
  * [Python interface](https://mailsort.readthedocs.io/en/latest/configuration.html#python-interface)
* [How mailsort works](https://mailsort.readthedocs.io/en/latest/architecture.html)
  * [The fetch-store-train-predict-move loop](https://mailsort.readthedocs.io/en/latest/architecture.html#the-fetch-store-train-predict-move-loop)
  * [What is stored, and where](https://mailsort.readthedocs.io/en/latest/architecture.html#what-is-stored-and-where)
* [Troubleshooting](https://mailsort.readthedocs.io/en/latest/troubleshooting.html)
* [Support](https://mailsort.readthedocs.io/en/latest/support.html)
* [Developer](https://mailsort.readthedocs.io/en/latest/developer.html)
  * [Python Interface](https://mailsort.readthedocs.io/en/latest/developer.html#python-interface)
  * [The mailsort.api module](https://mailsort.readthedocs.io/en/latest/developer.html#the-mailsort-api-module)

## Installation
```
pip install mailsort
```

## Command line interface
```
mailsort --host imap.example.com --username user@example.com --password "..." -u
mailsort --host imap.example.com --username user@example.com --password "..." -l "some_label"
```
The IMAP password is provided with `--password`, e.g. by having your shell pull it from a
password manager.

## Python interface
```python
from mailsort import Imap

imap = Imap(
    host="imap.example.com",
    port=993,
    username="user@example.com",
    password="...",
    connection_str="sqlite:///email.db",
)
imap.update_database(quick=False)
imap.fit_machine_learning_model_to_database()
imap.filter_messages_from_server(label="some_label", recommendation_ratio=0.9)
```

## API for downstream packages
Packages built on top of `mailsort`, such as `gmailsorter`, should import the shared database and
machine learning building blocks from `mailsort.api` rather than from mailsort's internal modules
directly. This keeps `mailsort.api` as the single place that needs to stay consistent when
mailsort's internals are refactored.
```python
from mailsort.api import (
    AbstractMailBox,
    AbstractMessage,
    DatabaseInterface,
    DatabaseTemplate,
    MachineLearningDatabase,
    email_date_converter,
    get_email_database,
    get_machine_learning_database,
    strip_html_tags,
)
```
