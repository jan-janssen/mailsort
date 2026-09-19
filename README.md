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
  * [Dry run / recommendation mode](https://mailsort.readthedocs.io/en/latest/configuration.html#dry-run-recommendation-mode)
  * [Python interface](https://mailsort.readthedocs.io/en/latest/configuration.html#python-interface)
* [How mailsort works](https://mailsort.readthedocs.io/en/latest/architecture.html)
  * [The fetch-store-train-predict-move loop](https://mailsort.readthedocs.io/en/latest/architecture.html#the-fetch-store-train-predict-move-loop)
  * [What is stored, and where](https://mailsort.readthedocs.io/en/latest/architecture.html#what-is-stored-and-where)
* [Troubleshooting](https://mailsort.readthedocs.io/en/latest/troubleshooting.html)
* [Support](https://mailsort.readthedocs.io/en/latest/support.html)
* [Developer](https://mailsort.readthedocs.io/en/latest/developer.html)
  * [Python Interface](https://mailsort.readthedocs.io/en/latest/developer.html#python-interface)
  * [Dry run / recommendation mode](https://mailsort.readthedocs.io/en/latest/developer.html#dry-run-recommendation-mode)
  * [The mailsort.api module](https://mailsort.readthedocs.io/en/latest/developer.html#the-mailsort-api-module)

## Installation
```
pip install mailsort
```

## Command line interface
`mailsort` is organized around five subcommands - `sync`, `train`, `predict`, `sort` and `status`:
```
mailsort sync --host imap.example.com --username user@example.com --password "..."
mailsort train
mailsort predict some_label
mailsort sort some_label
mailsort status
```
The IMAP password is provided with `--password`, e.g. by having your shell pull it from a
password manager.

- `sync` downloads new and changed messages from the mail server into the local database.
- `train` (re-)trains the machine learning model on the local database - it does not connect to
  the mail server, so it also works offline once `sync` has run at least once.
- `predict FOLDER` reports what the trained model would recommend for the messages currently in
  `FOLDER`, without moving, deleting or otherwise modifying anything on the server:
  ```
  MESSAGE ID                          SUBJECT                                  RECOMMENDED LABEL     SCORE  REACHED
  --------------------------------------------------------------------------------------------------------------
  some_label\x1f101                   Your invoice for March                  Receipts                1.00     True
  some_label\x1f102                   Let's catch up next week                -                       0.00    False
  ```
- `sort FOLDER` does the same scoring as `predict`, but actually moves the messages whose score
  clears the configured threshold (`--recommendation-ratio`, 90% by default).
- `status` reports the local database location, how many messages it knows about, and how many
  per-folder models have been trained - also without connecting to the mail server.

Run `mailsort --help` or `mailsort <command> --help` for the full list of options and examples.

### Upgrading from the pre-1.0 CLI
The previous flat `-u/--update` and `-l/--label` options still work exactly as before, but are
deprecated in favor of the subcommands above:
```
mailsort --host imap.example.com --username user@example.com --password "..." -u
mailsort --host imap.example.com --username user@example.com --password "..." -l "some_label"
mailsort --host imap.example.com --username user@example.com --password "..." -l "some_label" --dry-run
```
is equivalent to:
```
mailsort sync --host imap.example.com --username user@example.com --password "..."
mailsort train
mailsort sort some_label --host imap.example.com --username user@example.com --password "..."
mailsort predict some_label --host imap.example.com --username user@example.com --password "..."
```

## Python interface
The recommended way to use `mailsort` from Python is `MailSorter`, a small facade that wraps a
mailbox backend (such as `Imap`) and exposes the fetch-store-train-predict-move loop through the
same vocabulary as the CLI:
```python
from mailsort import Imap, MailSorter

with MailSorter(
    Imap(
        host="imap.example.com",
        port=993,
        username="user@example.com",
        password="...",
        connection_str="sqlite:///email.db",
    )
) as sorter:
    sorter.sync()
    sorter.train()
    predictions = sorter.predict("some_label")
    sorter.sort("some_label")
```
- `sync()` downloads new and changed messages into the local database and returns a `SyncResult`.
- `train()` (re-)trains the machine learning models on the local database and returns a
  `TrainResult`.
- `predict(folder)` is read-only: it downloads and scores the messages currently in `folder` and
  returns a `list[Prediction]`, without moving, deleting or otherwise modifying anything on the
  server.
- `sort(folder)` scores messages the same way as `predict()`, but actually moves the ones whose
  score clears `recommendation_ratio` (90% by default), and returns a `SortResult`.

`MailSorter` only relies on the public `AbstractMailBox` interface, not on anything IMAP-specific,
so it works unmodified with any current or future mailbox backend - including a Gmail backend
built by [gmailsorter](https://github.com/jan-janssen/gmailsorter).

### Dry run / recommendation mode
`predict()` computes the exact same machine learning recommendations `sort()` would act on, but
only returns them - it never moves, deletes, archives or otherwise modifies anything on the
server:
```python
for prediction in sorter.predict("some_label"):
    print(
        prediction.message_id,
        prediction.subject,
        prediction.recommended_label,
        prediction.score,
        prediction.threshold_reached,
    )
```
Each `Prediction` has:
- `message_id` - id that uniquely identifies the message
- `subject` - the message subject, if available
- `recommended_label` - the folder the model scores highest for this message, or `None` if no
  model has been trained yet
- `score` - the model's score for `recommended_label`
- `threshold_reached` - whether `score` clears `recommendation_ratio`, i.e. whether `sort()`
  would move this message for real

### Low-level interface
`MailSorter` is a thin wrapper around methods `Imap` (an `AbstractMailBox`) already exposes
directly. They remain available, both for backwards compatibility with existing scripts and for
callers who want finer-grained control - `MailSorter.sync()`/`train()`/`predict()`/`sort()` above
are implemented purely in terms of them, so behavior is identical either way:
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
`update_database()`, `fit_machine_learning_model_to_database()` and `filter_messages_from_server()`
return the same `SyncResult`/`TrainResult`/`SortResult` objects as their `MailSorter` counterparts.
`get_label_recommendations()`, the equivalent of `predict()`, returns a `list[dict]` rather than a
`list[Prediction]`, unchanged from earlier versions:
```python
for recommendation in imap.get_label_recommendations(
    label="some_label", recommendation_ratio=0.9
):
    print(recommendation["message_id"], recommendation["recommended_label"])
```

### Train and inspect the local database without a mail connection
`train_machine_learning_models()` and `get_database_status()` are the functions behind the
`train` and `status` CLI commands. Both only need the database connection string, not mail server
credentials:
```python
from mailsort.api import get_database_status, train_machine_learning_models

train_machine_learning_models(connection_str="sqlite:///email.db")
status = get_database_status(connection_str="sqlite:///email.db")
print(status.message_count, status.trained_label_lst)
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
    DatabaseStatus,
    DatabaseTemplate,
    MachineLearningDatabase,
    MailSorter,
    Prediction,
    SortResult,
    SyncResult,
    TrainResult,
    email_date_converter,
    get_database_status,
    get_email_database,
    get_machine_learning_database,
    strip_html_tags,
    train_machine_learning_models,
)
```
`MailSorter` and the `SyncResult`/`TrainResult`/`Prediction`/`SortResult` result types are exported
here, not just from `mailsort` directly, because they are generic over `AbstractMailBox`: a
downstream package implementing its own mailbox backend (as `gmailsorter` does for Gmail) can wrap
its own `AbstractMailBox` subclass in the same `MailSorter` facade without reimplementing it.
