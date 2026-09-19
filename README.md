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
* [Evaluation and confidence](https://mailsort.readthedocs.io/en/latest/evaluation.html)
  * [Why a classifier score is not automatically a probability](https://mailsort.readthedocs.io/en/latest/evaluation.html#why-a-classifier-score-is-not-automatically-a-probability)
  * [`mailsort evaluate`](https://mailsort.readthedocs.io/en/latest/evaluation.html#mailsort-evaluate)
* [Troubleshooting](https://mailsort.readthedocs.io/en/latest/troubleshooting.html)
* [Support](https://mailsort.readthedocs.io/en/latest/support.html)
* [Developer](https://mailsort.readthedocs.io/en/latest/developer.html)
  * [Python Interface](https://mailsort.readthedocs.io/en/latest/developer.html#python-interface)
  * [Dry run / recommendation mode](https://mailsort.readthedocs.io/en/latest/developer.html#dry-run-recommendation-mode)
  * [Evaluation](https://mailsort.readthedocs.io/en/latest/developer.html#evaluation)
  * [The mailsort.api module](https://mailsort.readthedocs.io/en/latest/developer.html#the-mailsort-api-module)

## Installation
```
pip install mailsort
```

## Command line interface
`mailsort` is organized around six subcommands - `sync`, `train`, `predict`, `sort`, `status` and `evaluate`:
```
mailsort sync --host imap.example.com --username user@example.com --password "..."
mailsort train
mailsort predict some_label
mailsort sort some_label
mailsort status
mailsort evaluate
```
The IMAP password is provided with `--password`, e.g. by having your shell pull it from a
password manager.

- `sync` downloads new and changed messages from the mail server into the local database.
- `train` (re-)trains the machine learning model on the local database - it does not connect to
  the mail server, so it also works offline once `sync` has run at least once. Each folder's
  model is calibrated automatically where there is enough data to do so safely - see
  [Evaluation and confidence](https://mailsort.readthedocs.io/en/latest/evaluation.html).
- `predict FOLDER` reports what the trained model would recommend for the messages currently in
  `FOLDER`, without moving, deleting or otherwise modifying anything on the server:
  ```
  MESSAGE ID                           SUBJECT                                  RECOMMENDED FOLDER    SCORE TYPE       ACCEPTED
  -----------------------------------------------------------------------------------------------------------------------------
  some_label\x1f101                     Your invoice for March                   Receipts               0.97 calibrated     True
  some_label\x1f102                     Let's catch up next week                 -                      0.00 raw           False
  ```
- `sort FOLDER` does the same scoring as `predict`, but actually moves the messages whose score
  clears the configured threshold (`--recommendation-ratio`, 90% by default). That threshold is a
  cutoff on a model score, not a guaranteed probability of being correct - use `mailsort evaluate`
  to check what it actually achieves on your own mailbox before trusting it.
- `status` reports the local database location, how many messages it knows about, and how many
  per-folder models have been trained - also without connecting to the mail server.
- `evaluate` estimates precision, recall, F1, support and coverage/abstention rate at
  `--recommendation-ratio` on held-out data - the evidence a threshold choice should be based on.
  Add `--sweep` to compare several thresholds at once. It never connects to the mail server and
  never changes the models `mailsort train` has already stored - see
  [Evaluation and confidence](https://mailsort.readthedocs.io/en/latest/evaluation.html).

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
`predict()` computes the exact same machine learning predictions `sort()` would act on, but only
returns them - it never moves, deletes, archives or otherwise modifies anything on the server.
Each message gets a `Prediction`, a plain, JSON-serializable dataclass - a first-class,
side-effect-free representation of one classification result, independent of any mailbox change:
```python
for prediction in sorter.predict("some_label"):
    print(
        prediction.message_id,
        prediction.source_folder,
        prediction.recommended_folder,
        prediction.score,
        prediction.threshold,
        prediction.accepted,
        prediction.score_type,
        prediction.subject,
    )
```
`Prediction` has:
- `message_id` - id that uniquely identifies the message
- `source_folder` - the folder the message was fetched and scored from
- `recommended_folder` - the folder the model scores highest for this message, or `None` if no
  model has been trained yet
- `score` - the model's score for `recommended_folder`
- `threshold` - the `recommendation_ratio` this prediction was scored against
- `accepted` - whether `score` clears `threshold`, i.e. whether `sort()` would move this message
  for real given the same `recommendation_ratio` - an abstained prediction (`accepted=False`) is
  never acted on
- `score_type` - whether `score` is a calibrated probability (`ScoreType.CALIBRATED`) or a raw,
  uncalibrated classifier score (`ScoreType.RAW`) - see
  [Evaluation and confidence](https://mailsort.readthedocs.io/en/latest/evaluation.html) for what
  that distinction means and why `recommendation_ratio` is not automatically a probability
- `subject` - the message subject, if available; display metadata, not itself part of the
  classification

Because every field is a plain value, `Prediction` is equally useful to the CLI's `predict` table,
to a caller such as `gmailsorter`, to a future web interface, or to an audit log - store or ship a
`Prediction` as-is, no scikit-learn objects involved.

### Evaluation
Before trusting `sort()`/`mailsort sort` to move mail automatically, check what a given
`recommendation_ratio` actually achieves on your own data, rather than assuming it does what the
number suggests:
```python
from mailsort.api import evaluate_models

report = evaluate_models(connection_str="sqlite:///email.db", recommendation_ratio=0.9)
print(report.coverage, report.overall_precision)
```
This trains a separate, throwaway set of models on part of your local database and scores them
against the rest - it never touches the models `mailsort train` has already stored. See
[Evaluation and confidence](https://mailsort.readthedocs.io/en/latest/evaluation.html) for the
full reasoning (why a classifier score is not automatically a probability, when calibration is and
is not applied, and how to read the report), and the command line equivalent,
`mailsort evaluate`.

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
`get_label_recommendations()` is `predict()`'s implementation - it already returns `list[Prediction]`,
so `MailSorter.predict()` is a pure passthrough to it:
```python
for prediction in imap.get_label_recommendations(
    label="some_label", recommendation_ratio=0.9
):
    print(prediction.message_id, prediction.recommended_folder)
```
`filter_messages_from_server()` scores messages through this exact same call, then moves only the
accepted predictions - inference and mailbox mutation are separated in code, not just by
convention, so it is not possible to move a message without it first having gone through
`get_label_recommendations()`.

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
    EvaluationReport,
    FolderMetrics,
    MachineLearningDatabase,
    MailSorter,
    Prediction,
    ScoreType,
    SortResult,
    SyncResult,
    TrainResult,
    email_date_converter,
    evaluate_models,
    get_database_status,
    get_email_database,
    get_machine_learning_database,
    strip_html_tags,
    train_machine_learning_models,
)
```
`MailSorter`, `evaluate_models()` and the `SyncResult`/`TrainResult`/`Prediction`/`SortResult`/
`EvaluationReport`/`FolderMetrics` result types are exported here, not just from `mailsort`
directly, because they are generic over `AbstractMailBox`: a downstream package implementing its
own mailbox backend (as `gmailsorter` does for Gmail) can wrap its own `AbstractMailBox` subclass
in the same `MailSorter` facade, and evaluate its own database the same way, without
reimplementing either.
