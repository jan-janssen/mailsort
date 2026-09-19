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
- `train(n_estimators=100, max_features=400, random_state=42, bootstrap=True, include_deleted=False, max_workers=None, calibrate=True, min_samples_per_class_for_calibration=20, max_calibration_cv_folds=5)`
  - (re-)train one machine learning model per folder on the local database; returns a `TrainResult` with
  `trained_label_lst` and `model_count`. Each folder's classifier is calibrated when (and only when) it has enough
  training data to do so safely - see [Evaluation and confidence](evaluation) and `mailsort.ml.calibration` below;
  pass `calibrate=False` to keep every folder's raw score instead.
- `predict(folder, recommendation_ratio=0.9, label_prefix="labels_")` - read-only; returns a `list[Prediction]`
  without moving, deleting or otherwise modifying anything on the server. See "Dry run / recommendation mode"
  below.
- `sort(folder, recommendation_ratio=0.9, label_prefix="labels_")` - scores messages exactly as `predict()` does
  with the same arguments, then moves the ones that clear `recommendation_ratio`; returns a `SortResult` with
  `moved_lst` and `moved_count`. This is the only `MailSorter` method that changes the mailbox.
  `recommendation_ratio` is a cutoff on a model score, not a guaranteed probability of being correct - see
  [Evaluation and confidence](evaluation) for what it actually means and how to check it against your own data
  before trusting it, with `mailsort.evaluation`/`mailsort evaluate`.

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
database and tries to predict the correct folder for these emails. `recommendation_ratio` is the cutoff a model
score must clear to actually move the email - a threshold on a classifier score, not a guaranteed probability of
being correct (see [Evaluation and confidence](evaluation) for what the score actually is and how to check what a
given threshold achieves on your own data with `mailsort.evaluation`/`mailsort evaluate`). It returns a
`SortResult` with the `(message_id, label)` pairs actually moved (`moved_lst`) and how many that is
(`moved_count`).

### Calibration
Where a folder has enough training data, `fit_machine_learning_model_to_database()` (and `MailSorter.train()`
above) also calibrates that folder's classifier - rescaling its scores against held-out data so a score more
honestly reflects how often it is actually right, rather than leaving it as a raw, uncalibrated classifier score.
See [Evaluation and confidence](evaluation) for the full reasoning; in short, `mailsort.ml.calibration`:
```python
from mailsort.ml.calibration import is_calibrated, should_calibrate

should_calibrate(y, min_samples_per_class=20)  # y: a folder's binary (0.0/1.0) training target
```
decides per folder whether there is enough data to calibrate safely (both classes need at least
`min_samples_per_class_for_calibration` examples, 20 by default), and `is_calibrated(model)` tells you which kind
of model you got back. Every `Prediction` also carries this as `score_type` (`ScoreType.CALIBRATED` or
`ScoreType.RAW`), so the distinction is explicit wherever a score is used, not just at training time:
```python
from mailsort import ScoreType

for prediction in sorter.predict("MailSortInbox"):
    if prediction.score_type is ScoreType.RAW:
        print(f"{prediction.message_id}: uncalibrated score, treat with more caution")
```

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
- `score_type` (`ScoreType`) - whether `score` is a calibrated probability (`ScoreType.CALIBRATED`) or a raw,
  uncalibrated classifier score (`ScoreType.RAW`) - see [Evaluation and confidence](evaluation) and "Calibration"
  above for what that distinction means and why it is never implicit.
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

## Evaluation
`recommendation_ratio` is a threshold on a model score, not a guaranteed probability of being correct - see
[Evaluation and confidence](evaluation) for the full reasoning. `mailsort.api.evaluate_models()` is the Python
equivalent of `mailsort evaluate`: it trains a separate, throwaway set of per-folder models on part of the local
database and scores them against the rest, without touching the models `mailsort train` has already stored:
```python
from mailsort.api import evaluate_models

report = evaluate_models(connection_str="sqlite:///email.db", recommendation_ratio=0.9)
print(report.coverage, report.overall_precision)
for folder_metrics in report.folder_metrics:
    print(
        folder_metrics.folder,
        folder_metrics.support,
        folder_metrics.precision,
        folder_metrics.recall,
        folder_metrics.f1,
    )
```
`report` is an `EvaluationReport` - a frozen, JSON-serializable dataclass, like `Prediction` - with:
- `coverage` (`float`) - fraction of held-out test messages accepted at `recommendation_ratio`.
- `overall_precision` (`float`/`None`) - among *accepted* predictions only, how often the recommended folder was
  actually correct; `None` if nothing was accepted. See [Evaluation and confidence](evaluation) for why this is
  the number to weigh most heavily when picking a threshold.
- `folder_metrics` (`list[FolderMetrics]`) - `precision`/`recall`/`f1`/`support`, plus `true_positive`/
  `false_positive`/`misrouted`/`abstained` counts, per folder.
- `confusion` (`dict[str, dict[str, int]]`) - `confusion[true_folder][recommended_folder]` counts, over accepted
  predictions only.
- `calibration_status` (`dict[str, bool]`) - whether each folder's held-out classifier ended up calibrated.
- `train_message_count`/`test_message_count`/`excluded_message_count` - so you can judge how much to trust a
  given report; see the limitations in [Evaluation and confidence](evaluation).

To compare several thresholds without retraining for each one (what `mailsort evaluate --sweep` does), use the
lower-level `mailsort.evaluation`/`mailsort.ml.evaluation` functions directly:
```python
from mailsort.evaluation import evaluate_at_threshold, score_holdout

holdout = score_holdout(connection_str="sqlite:///email.db")  # trains and scores once
for ratio in (0.7, 0.8, 0.9, 0.95):
    report = evaluate_at_threshold(holdout, recommendation_ratio=ratio)
    print(ratio, report.coverage, report.overall_precision)
```

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
`MailSorter` and the result types are exported here specifically because they are generic over `AbstractMailBox`:
a downstream package's own mailbox backend (a `gmailsorter` Gmail backend, for example) is itself an
`AbstractMailBox` subclass, so it can be wrapped in this same `MailSorter` without any changes. The same applies to
`evaluate_models()`/`EvaluationReport`/`FolderMetrics` - evaluation only needs the local database, not anything
IMAP-specific, so it works the same way for a `gmailsorter` Gmail-backed database too.

Prefer importing from `mailsort.api` over `mailsort`'s internal modules (`mailsort.base.*`, `mailsort.ml.*`) when
integrating with `mailsort` from another package, since `mailsort.api` is kept consistent across refactors while
the internal module layout is not.

## Future directions
The current machine learning model is limited in precision and memory usage. So there is a great interest to
replace it with a computationally more efficient model. All suggestions and feedback are welcome. Beyond the
optimization of the machine learning model and general improvements to the stability of the code base, adding
built-in scheduling would be a natural next step, though it is currently on hold based on limited resources.
