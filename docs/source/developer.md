# Developer
To simplify integration into other projects, all the functionality of `mailsort` is available as a plain Python
module - the command line interface described in [Configuration](configuration) is a thin wrapper around it.
[gmailsorter](https://github.com/jan-janssen/gmailsorter) builds its Gmail-specific support on top of the same
shared IMAP and machine learning core.

## Python Interface
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
ignored.

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
This is the equivalent of `mailsort sort`/`mailsort predict`'s `Imap` object retraining. If you only need to
(re-)train from an already-synced local database - the `mailsort train` CLI command's use case - use
`train_machine_learning_models()` instead, which does not need a mail server connection at all:
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
certainty required to actually move the email, with `0.9` equalling a certainty of 90%.

## Dry run / recommendation mode
`get_label_recommendations()` runs the exact same download and scoring steps as
`filter_messages_from_server()`, but only returns the result instead of acting on it - it never moves, deletes,
archives or otherwise modifies anything on the server, so it is safe to call at any time, including before you
trust the model with your mailbox:
```
recommendations = imap.get_label_recommendations(
    label="MailSortInbox",
    recommendation_ratio=0.9,
    label_prefix="labels_",
)
```
`recommendations` is a list with one dict per message currently in `"MailSortInbox"`, each with:
- `message_id` (`str`) - id that uniquely identifies the message.
- `subject` (`str`/`None`) - the message subject, if available.
- `recommended_label` (`str`/`None`) - the folder the model scores highest for this message, or `None` if no
  machine learning model has been trained yet (run `fit_machine_learning_model_to_database()` first).
- `score` (`float`) - the model's score for `recommended_label`.
- `threshold_reached` (`bool`) - whether `score` clears `recommendation_ratio`, i.e. whether
  `filter_messages_from_server()` would move this particular message for real, given the same
  `recommendation_ratio`.

For example, to only print the messages that would actually be moved:
```
for recommendation in recommendations:
    if recommendation["threshold_reached"]:
        print(
            f"{recommendation['subject']!r} -> {recommendation['recommended_label']} "
            f"(score {recommendation['score']:.2f})"
        )
```
The command line equivalent is `mailsort predict MailSortInbox` - see [Configuration](configuration).

## The mailsort.api module
`mailsort.api` re-exports the building blocks (database helpers, the abstract mailbox and message base classes,
and the machine learning helpers) that a package building its own mailbox integration on top of `mailsort` - such
as `gmailsorter` - needs, without depending on `mailsort`'s internal module layout directly:
```
from mailsort.api import (
    AbstractMailBox,
    AbstractMessage,
    DatabaseInterface,
    DatabaseStatus,
    DatabaseTemplate,
    MachineLearningDatabase,
    email_date_converter,
    get_database_status,
    get_email_database,
    get_machine_learning_database,
    strip_html_tags,
    train_machine_learning_models,
)
```
Prefer importing from `mailsort.api` over `mailsort`'s internal modules (`mailsort.base.*`, `mailsort.ml.*`) when
integrating with `mailsort` from another package, since `mailsort.api` is kept consistent across refactors while
the internal module layout is not.

## Future directions
The current machine learning model is limited in precision and memory usage. So there is a great interest to
replace it with a computationally more efficient model. All suggestions and feedback are welcome. Beyond the
optimization of the machine learning model and general improvements to the stability of the code base, adding
built-in scheduling would be a natural next step, though it is currently on hold based on limited resources.
