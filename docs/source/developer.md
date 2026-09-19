# Developer
To simplify integration into other projects, all the functionality of `mailsort` is available as a plain Python
module - the command line interface described in [Configuration](configuration) is a thin wrapper around it.
[gmailsorter](https://github.com/jan-janssen/gmailsorter) builds its Gmail-specific support on top of the same
shared IMAP and machine learning core.

## MailSorter - the recommended interface
`MailSorter` is a small facade over a mailbox backend (such as `Imap`) that exposes the fetch-store-train-predict-
move loop through the same task-oriented vocabulary as the CLI - `sync`/`train`/`predict`/`sort` - instead of the
lower-level `update_database()`/`fit_machine_learning_model_to_database()`/`filter_messages_from_server()` methods
those tasks are built from:
```python
from mailsort import Imap, MailSorter

with MailSorter(
    Imap(
        host="imap.example.com",
        port=993,
        username="user@example.com",
        password="app-password",
        connection_str="sqlite:///email.db",
    )
) as sorter:
    sync_result = sorter.sync()
    train_result = sorter.train()
    predictions = sorter.predict("MailSortInbox")
    sort_result = sorter.sort("MailSortInbox")
```
- `sync(quick=False, label_lst=None, email_format=None)` - update the local database from the mail server; returns
  a `SyncResult` with `new_message_count`, `updated_message_count` and `deleted_message_count`.
- `train(n_estimators=100, max_features=400, random_state=42, bootstrap=True, include_deleted=False, max_workers=None)`
  - (re-)train one machine learning model per folder on the local database; returns a `TrainResult` with
  `trained_label_lst` and `model_count`.
- `predict(folder, recommendation_ratio=0.9, label_prefix="labels_")` - read-only; returns a `list[Prediction]`
  without moving, deleting or otherwise modifying anything on the server. See "Dry run / recommendation mode"
  below.
- `sort(folder, recommendation_ratio=0.9, label_prefix="labels_")` - scores messages exactly as `predict()` does
  with the same arguments, then moves the ones that clear `recommendation_ratio`; returns a `SortResult` with
  `moved_lst` and `moved_count`. This is the only `MailSorter` method that changes the mailbox.

`MailSorter` itself is a context manager - entering it returns the `MailSorter`, and exiting it (or calling
`sorter.close()` directly) closes the wrapped mailbox, exactly like using `Imap` as a context manager directly.

`MailSorter` only calls the public `AbstractMailBox` interface - `close()`, `update_database()`,
`fit_machine_learning_model_to_database()`, `get_label_recommendations()` and `filter_messages_from_server()` - so
it works unmodified with any `AbstractMailBox` subclass, not just `Imap`. A package building its own mailbox
backend on top of `mailsort` (as `gmailsorter` does for Gmail) can wrap that backend in the same `MailSorter`
without reimplementing it - see "The mailsort.api module" below.

## Low-level interface
`MailSorter` above is implemented purely in terms of the methods described in this section - `Imap` (and any other
`AbstractMailBox` subclass) already exposes the full fetch-store-train-predict-move loop directly. They remain
available for backwards compatibility with existing scripts, and for callers who want finer-grained control than
`MailSorter` provides.

Just install the `mailsort` python package and then import the `Imap` class from the `mailsort` module:
```
from mailsort import Imap
```

### Initialize mailsort
Create an `imap` object from the `Imap()` class:
```
imap = Imap(
    host="imap.example.com",
    port=993,
    username="user@example.com",
    password="app-password",
    connection_str="sqlite:///email.db",
)
```
- `host`/`port` are the hostname and port of your IMAP server, typically `993` for `IMAP4_SSL` (the default) or
  `143` for plain `IMAP4` (set `use_ssl=False` in that case).
- `username`/`password` are your IMAP account credentials, for example an app password - see
  [Preparation](preparation).
- `connection_str` is a connection to an SQL database, provided as an [SQLAlchemy](https://www.sqlalchemy.org/)
  connection string.
- `db_user_id` (default `1`) lets multiple accounts share the same database without mixing their data.

The IMAP connection is kept open for the lifetime of the `Imap` object. Call `imap.close()` when you are done with
it, or use it as a context manager instead:
```
with Imap(
    host="imap.example.com",
    port=993,
    username="user@example.com",
    password="app-password",
    connection_str="sqlite:///email.db",
) as imap:
    imap.update_database(quick=False)
```

### Sync local database with email account
To reduce the communication overhead, the emails are stored locally in an SQLite database:
```
imap.update_database(quick=False)
```
By setting the optional flag `quick` to `True` only new emails are downloaded while changes to existing emails are
ignored. It returns a `SyncResult` (`new_message_count`, `updated_message_count`, `deleted_message_count`) -
`updated_message_count` and `deleted_message_count` are always `0` when `quick=True`, since that mode skips both
steps.

### Generate pandas dataframe for emails
Load all emails from the local database and combine them in a pandas `DataFrame` for further postprocessing:
```
df = imap.get_all_emails_in_database()
```

### Download a specific folder from the email server
Download emails currently in the folder `"MyFolder"` from the email server:
```
df = imap.download_emails_for_label(label="MyFolder")
```
In this case the emails are not stored in the local database.

### Train the machine learning model
Train one machine learning model per folder on the emails currently stored in the local database:
```
imap.fit_machine_learning_model_to_database(
    n_estimators=100,
    max_features=400,
    random_state=42,
    bootstrap=True,
    include_deleted=False,
)
```
It returns a `TrainResult` with the sorted list of folders a model was trained for (`trained_label_lst`) and how
many that is (`model_count`).

If you only need to (re-)train from an already-synced local database and do not have an `Imap`/`AbstractMailBox`
object at hand - the `mailsort train` CLI command's use case - use `train_machine_learning_models()` instead, which
does not need a mail server connection at all:
```python
from mailsort.api import train_machine_learning_models

train_machine_learning_models(
    connection_str="sqlite:///email.db",
    n_estimators=100,
    max_features=400,
    random_state=42,
    bootstrap=True,
    include_deleted=False,
)
```

### Inspect the local database
`get_database_status()` reports the same information as the `mailsort status` CLI command - the database
location, how many messages it knows about, and which folders a model has been trained for - also without
connecting to the mail server:
```python
from mailsort.api import get_database_status

status = get_database_status(connection_str="sqlite:///email.db")
print(status.message_count, status.active_message_count, status.deleted_message_count)
print(status.trained_label_lst, status.feature_count)
```

### Filter emails using machine learning
Assign new emails in the folder `"MailSortInbox"` to the folder that best matches them:
```
imap.filter_messages_from_server(
    label="MailSortInbox",
    recommendation_ratio=0.9,
    label_prefix="labels_",
)
```
It checks the server for new emails in the given folder, reloads the machine learning models from the local
database and tries to predict the correct folder for these emails. The `recommendation_ratio` defines the level of
certainty required to actually move the email, with `0.9` equalling a certainty of 90%. It returns a `SortResult`
with the `(message_id, label)` pairs actually moved (`moved_lst`) and how many that is (`moved_count`).

## Dry run / recommendation mode
`get_label_recommendations()` runs the exact same download and scoring steps as
`filter_messages_from_server()`, but only returns the result instead of acting on it - it never moves, deletes,
archives or otherwise modifies anything on the server, so it is safe to call at any time, including before you
trust the model with your mailbox. It is `MailSorter.predict()`'s implementation - `predict()` is a pure
passthrough to it - so both already return the same `list[Prediction]`:
```
predictions = imap.get_label_recommendations(
    label="MailSortInbox",
    recommendation_ratio=0.9,
    label_prefix="labels_",
)
```
Each `Prediction` is a frozen, JSON-serializable dataclass - a first-class, side-effect-free representation of one
classification result - with:
- `message_id` (`str`) - id that uniquely identifies the message.
- `source_folder` (`str`) - the folder the message was fetched and scored from (`"MailSortInbox"` above).
- `recommended_folder` (`str`/`None`) - the folder the model scores highest for this message, or `None` if no
  machine learning model has been trained yet (run `fit_machine_learning_model_to_database()` first).
- `score` (`float`) - the model's score for `recommended_folder`.
- `threshold` (`float`) - the `recommendation_ratio` this prediction was scored against, carried alongside `score`
  so a `Prediction` is self-contained.
- `accepted` (`bool`) - whether `score` clears `threshold`, i.e. whether `filter_messages_from_server()` would
  move this particular message for real, given the same `recommendation_ratio`. An abstained prediction
  (`accepted=False`) is never acted on.
- `subject` (`str`/`None`) - the message subject, if available - display metadata, not itself part of the
  classification.

`filter_messages_from_server()` scores messages through this exact same call and then moves only the accepted
predictions - inference and mailbox mutation are separated in code, not just by convention, so a message can only
be moved after having gone through `get_label_recommendations()` first. This also means
`filter_messages_from_server()` inherits `get_label_recommendations()`'s safe handling of an untrained database
(every message abstains rather than the machine learning pipeline running against zero models).

For example, to only print the messages that would actually be moved:
```
for prediction in predictions:
    if prediction.accepted:
        print(f"{prediction.subject!r} -> {prediction.recommended_folder} (score {prediction.score:.2f})")
```
The command line equivalent is `mailsort predict MailSortInbox` - see [Configuration](configuration).

## The mailsort.api module
`mailsort.api` re-exports the building blocks (database helpers, the abstract mailbox and message base classes,
the machine learning helpers, `MailSorter` and its result types) that a package building its own mailbox
integration on top of `mailsort` - such as `gmailsorter` - needs, without depending on `mailsort`'s internal module
layout directly:
```
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
`MailSorter` and the result types are exported here specifically because they are generic over `AbstractMailBox`:
a downstream package's own mailbox backend (a `gmailsorter` Gmail backend, for example) is itself an
`AbstractMailBox` subclass, so it can be wrapped in this same `MailSorter` without any changes.

Prefer importing from `mailsort.api` over `mailsort`'s internal modules (`mailsort.base.*`, `mailsort.ml.*`) when
integrating with `mailsort` from another package, since `mailsort.api` is kept consistent across refactors while
the internal module layout is not.

## Future directions
The current machine learning model is limited in precision and memory usage. So there is a great interest to
replace it with a computationally more efficient model. All suggestions and feedback are welcome. Beyond the
optimization of the machine learning model and general improvements to the stability of the code base, adding
built-in scheduling would be a natural next step, though it is currently on hold based on limited resources.
