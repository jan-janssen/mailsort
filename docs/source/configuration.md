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
Once installed, `mailsort` is also available as a command line tool. To (re-)train the machine learning model on
your existing folders:
```
mailsort --host imap.example.com --username user@example.com --password "..." -d sqlite:///email.db -u
```
And to sort new emails from the `MailSortInbox` folder created in [Preparation](preparation) using the trained
model:
```
mailsort --host imap.example.com --username user@example.com --password "..." -d sqlite:///email.db -l MailSortInbox
```
The available command line options are:
- `--host` IMAP server hostname, e.g. `imap.example.com`.
- `--port` IMAP server port - default: `993`.
- `--username` IMAP account username.
- `--password` IMAP account password, for example an app password - see [Preparation](preparation).
- `--no-ssl` connect without SSL (IMAP4 instead of IMAP4_SSL).
- `-d/--database` connection string to connect to the database, e.g. `sqlite:///email.db`. By default `mailsort`
  uses a simple SQLite database, but most SQL databases supported by [SQLAlchemy](https://www.sqlalchemy.org/) work.
- `-u/--update` update the local email database and retrain the machine learning model.
- `-l/--label` email folder to be filtered with machine learning, e.g. `MailSortInbox`.
- `-i/--identification` user id of the database user, useful when sharing one database between multiple accounts -
  default: `1`.

Because `--password` accepts the password as a plain argument, prefer having your shell pull it from a password
manager (for example `--password "$(pass show imap/example.com)"`) rather than typing it directly, so it does not
end up in your shell history in plain text.

`mailsort` does not include its own scheduler, so to sort emails automatically every few minutes, schedule the
second command above with `cron` or a similar tool on your own machine.

## Python interface
To integrate `mailsort` into your own scripts, or to build additional functionality on top of it (as
[gmailsorter](https://github.com/jan-janssen/gmailsorter) does), see the [Developer](developer) page for the Python
API.
