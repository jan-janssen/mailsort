# Configuration
`mailsort` is a plain Python package - there is no hosted service, docker container or daemon to configure. You
install it, provide your IMAP account details, and run it yourself, either from the command line or from Python.
Since your account credentials and local database never leave the machine you run `mailsort` on, no data is shared
with any third party.

## Install mailsort
The `mailsort` package is available on PyPI and can be installed with:
```
pip install mailsort
```

## Command line interface
Once installed, `mailsort` is also available as a command line tool, built around six subcommands: `sync`,
`train`, `predict`, `sort`, `status` and `evaluate`.

First, download your existing folders into the local database:
```
mailsort sync --host imap.example.com --username user@example.com --password "..." -d sqlite:///email.db
```
Then train the machine learning model on them - this only reads the local database, so it does not need `--host`,
`--username` or `--password`:
```
mailsort train -d sqlite:///email.db
```
And to sort new emails from the `MailSortInbox` folder created in [Preparation](preparation) using the trained
model:
```
mailsort sort MailSortInbox --host imap.example.com --username user@example.com --password "..." -d sqlite:///email.db
```
`mailsort` does not include its own scheduler, so to sort emails automatically every few minutes, schedule the
command above with `cron` or a similar tool on your own machine.

### Scheduling with cron
A common setup is to run `mailsort sort` every 5 minutes to keep `MailSortInbox` clear, and `mailsort sync` followed
by `mailsort train` once a day to pick up messages you filed manually and keep the model up to date with them. Add
something like this to your crontab (`crontab -e`), pulling the password from a password manager rather than storing
it in the crontab in plain text:
```cron
# Sort new mail every 5 minutes
*/5 * * * * mailsort sort MailSortInbox --host imap.example.com --username user@example.com --password "$(pass show imap/example.com)" -d sqlite:///home/user/email.db

# Sync and retrain once a day at 03:00
0 3 * * * mailsort sync --host imap.example.com --username user@example.com --password "$(pass show imap/example.com)" -d sqlite:///home/user/email.db && mailsort train -d sqlite:///home/user/email.db
```
Use an absolute path for the SQLite database (as above), since `cron` jobs do not run with the working directory you
run `mailsort` from interactively. If `mailsort` is installed in a virtual environment, use the full path to its
executable (e.g. `/home/user/.venv/bin/mailsort`) instead of just `mailsort`, since `cron` runs with a minimal `PATH`.

To check what `mailsort` currently knows - where the database lives, how many messages it has stored, and how
many per-folder models have been trained:
```
mailsort status -d sqlite:///email.db
```

Before trusting `mailsort sort` to move mail automatically, check how well it is likely to do that on your own
data - `recommendation_ratio` (used by `predict`/`sort` below) is a threshold on a model score, not a guaranteed
error rate, see [Evaluation and confidence](evaluation):
```
mailsort evaluate -d sqlite:///email.db
```

Run `mailsort --help` for the full list of top-level options, or `mailsort <command> --help` (e.g.
`mailsort sync --help`) for the options of an individual command, including usage examples.

The connection-related options shared by `sync`, `predict` and `sort` are:
- `--host` IMAP server hostname, e.g. `imap.example.com`.
- `--port` IMAP server port - default: `993`.
- `--username` IMAP account username.
- `--password` IMAP account password, for example an app password - see [Preparation](preparation).
- `--no-ssl` connect without SSL (IMAP4 instead of IMAP4_SSL).

Available on every command:
- `-d/--database` connection string to connect to the database, e.g. `sqlite:///email.db`. By default `mailsort`
  uses a simple SQLite database, but most SQL databases supported by [SQLAlchemy](https://www.sqlalchemy.org/) work.
- `-i/--identification` user id of the database user, useful when sharing one database between multiple accounts -
  default: `1`.

Because `--password` accepts the password as a plain argument, prefer having your shell pull it from a password
manager (for example `--password "$(pass show imap/example.com)"`) rather than typing it directly, so it does not
end up in your shell history in plain text.

### Command reference
- `mailsort sync` downloads new and changed messages into the local database. Add `--quick` to only fetch new
  messages and skip re-checking the folders of already-known ones, or `--folder FOLDER` (repeatable) to restrict
  the sync to specific folders instead of every folder.
- `mailsort train` (re-)trains one machine learning model per folder on the messages already stored in the local
  database. It never connects to the mail server. Add `--include-deleted` to also train on messages marked as
  deleted, or `--no-calibration` to keep every folder's raw classifier score even where there would be enough data
  to calibrate it (see [Evaluation and confidence](evaluation) - calibration is applied automatically per folder
  where there is enough data, this only turns it off entirely).
- `mailsort predict FOLDER` downloads the messages currently in `FOLDER` and reports the model's recommendation
  for each, without moving, deleting or otherwise modifying anything on the server - see "Dry run / recommendation
  mode" below.
- `mailsort sort FOLDER` does the same scoring as `predict`, but moves the messages whose score clears
  `--recommendation-ratio` to the recommended folder.
- `mailsort status` reports the database location, the number of known messages (and how many of those are marked
  deleted), and the folders a model has been trained for. It never connects to the mail server.
- `mailsort evaluate` estimates precision, recall, F1, support and coverage/abstention rate at
  `--recommendation-ratio` on held-out data, so a threshold choice is evidence-based rather than assumed - see
  [Evaluation and confidence](evaluation) for the full explanation. It never connects to the mail server, and never
  changes the models `mailsort train` has already stored. Accepts `--test-size` (fraction of email threads held
  out for testing, default `0.25`), `--no-calibration`, and `--sweep` to compare several thresholds
  (`0.5`/`0.7`/`0.8`/`0.9`/`0.95`/`0.99`) at once instead of reporting a single one.

`predict` and `sort` additionally accept:
- `--recommendation-ratio` cutoff a score must clear to count as a recommendation (`0<r<1`) - default: `0.9`. This
  is a threshold on a model score, not a guaranteed probability of being correct - see
  [Evaluation and confidence](evaluation).
- `--label-prefix` prefix used to recognise label columns during feature encoding - only needs to change if you
  customised this in the Python API - default: `labels_`.

### Exit codes
`mailsort` returns `0` on success, `2` for invalid configuration (e.g. a command that needs `--host`/`--username`/
`--password` and does not have them, or a non-numeric `-i/--identification`), and `1` if an error is raised while
running the command (e.g. the mail server rejects the connection).

## Dry run / recommendation mode
Before letting `mailsort` move emails automatically, or when you simply want to see what the model would do with a
folder without touching your mailbox, use `mailsort predict` instead of `mailsort sort`:
```
mailsort predict MailSortInbox --host imap.example.com --username user@example.com --password "..." -d sqlite:///email.db
```
This downloads and scores the messages in `MailSortInbox` exactly as `mailsort sort` would, and prints the result
as a table - but it never moves, deletes, archives or otherwise modifies anything on the server:
```
MESSAGE ID                           SUBJECT                                  RECOMMENDED FOLDER    SCORE TYPE       ACCEPTED
-----------------------------------------------------------------------------------------------------------------------------
MailSortInbox\x1f101                  Your invoice for March                   Receipts               0.97 calibrated     True
MailSortInbox\x1f102                  Let's catch up next week                 -                      0.00 raw           False
```
Each row shows one message currently in the folder: its id, its subject, the folder the model would move it to,
the model's score for that folder, whether that score is a calibrated probability or a raw, uncalibrated
classifier score (`TYPE` - see [Evaluation and confidence](evaluation)), and whether the score clears
`--recommendation-ratio` (90% by default) - i.e. whether running `mailsort sort` on the same folder would actually
move that message. A `-` in the recommended folder column means either no folder scored high enough, or no machine
learning model has been trained yet (run `mailsort train` first).

## Python interface
To integrate `mailsort` into your own scripts, or to build additional functionality on top of it (as
[gmailsorter](https://github.com/jan-janssen/gmailsorter) does), see the [Developer](developer) page for the Python
API.
