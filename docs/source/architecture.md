# How mailsort works
This page explains what actually happens behind the scenes when `mailsort` sorts your emails. It is written for
curious users who want to understand the mechanics before trusting a program to move their emails around - no prior
machine learning knowledge required. If you are looking for setup instructions instead, see
[Preparation](preparation) and [Configuration](configuration). If you are looking for the Python API, see
[Developer](developer).

## The big picture
`mailsort` is built from three building blocks:

* **Your email account** - the source of truth for your emails and folders, accessed through a plain IMAP
  connection. `mailsort` never stores your account password, it is only used to log in and is not persisted
  anywhere. Each IMAP folder plays the role a label plays in a system like Gmail - "moving" an email between
  folders is the unit of sorting `mailsort` performs.
* **A local database** - a SQL database (SQLite by default, though any database supported by
  [SQLAlchemy](https://www.sqlalchemy.org/) works) that keeps a private copy of your email metadata and your
  trained models. This database lives entirely on your own machine.
* **One machine learning model per folder** - a small classifier trained purely on your own historic emails, which
  is used to recommend a folder for each new email that arrives in your sorting inbox folder (for example
  `MailSortInbox`).

Unlike a hosted service, `mailsort` does not run a background job for you - you trigger the loop below yourself,
either manually from the command line or on a schedule with `cron` (see [Configuration](configuration)).

## The fetch-store-train-predict-move loop

### 1. Fetch
`mailsort` asks the IMAP server for the list of message IDs that currently sit in a given folder (for example every
folder you already sorted your inbox into, or - during regular runs - your sorting inbox folder). Only message IDs
and metadata are requested at this stage, so this step is cheap and fast even for large mailboxes.

### 2. Store
For every message ID that is not yet known locally, `mailsort` downloads the message and splits it into several
normalized tables in the local database: the message content and subject, its thread, its `to`/`cc`/`from`
addresses and the folder it currently sits in. Messages that disappear from the server (for example because you
deleted them) are marked as deleted rather than removed, so your training history stays intact. This mirrors how a
relational database would model any many-to-many relationship - one email can have many recipients and many
folders it has passed through, so each of those gets its own table linked back to the email by its ID.

### 3. Train
This is the step that is easy to get wrong when described casually, so let's be precise: **mailsort does not read
the text of your emails to judge similarity.** Instead it looks at the metadata surrounding each email you have
already sorted:

* who sent it, and the domain of the sender (e.g. `@newsletter.example.com`),
* everyone it was addressed to or copied on,
* which email thread it belongs to,
* and which folder you filed it in.

Each of these values is turned into a binary "yes/no" column through
[one-hot encoding](https://en.wikipedia.org/wiki/One-hot) (implemented in
[`src/mailsort/ml/encoding.py`](https://github.com/jan-janssen/mailsort/blob/main/src/mailsort/ml/encoding.py)) -
so instead of one column "sender", you get one column per sender that is either `1` or `0`. `mailsort` then trains
one [random forest classifier](https://en.wikipedia.org/wiki/Random_forest) per folder
(see [`src/mailsort/ml/model.py`](https://github.com/jan-janssen/mailsort/blob/main/src/mailsort/ml/model.py)) to
answer the question "does this email belong to folder X, based on who sent it, who else received it and what other
folders tend to go together?". In practice this captures the intuition most people actually sort emails by - the
sender, the mailing list or the group of people involved - rather than trying to summarize free text.

The trained models are serialized and stored back in your local database, together with the exact list of columns
they were trained on, so they can be reloaded without retraining every time.

### 4. Predict
When a new email lands in your sorting inbox folder, `mailsort` encodes it using the very same columns the models
were trained on and asks every stored model "how confident are you that this email belongs to your folder?". The
folder with the highest confidence wins, but only if that confidence clears the `recommendation_ratio` threshold
(90% by default). If no folder is confident enough, the email is simply left where it is until the next run, once
more training data has made the models more confident.

### 5. Move
If a folder was recommended with sufficient confidence, `mailsort` issues an IMAP command to move the email out of
the sorting inbox folder and into the recommended folder - the same action you would take by hand by dragging the
email into a folder. Nothing is deleted or archived silently; the email simply moves to the folder you would have
put it in yourself.

To avoid ever recommending a folder you would not want emails moved into automatically, `mailsort` skips folders
marked with `\Noselect`, and special-use folders such as Trash, Spam/Junk, Sent and Drafts, both when training and
when predicting.

If you want to see steps 1-4 of this loop without ever letting `mailsort` perform step 5, see the dry run /
recommendation mode described in [Configuration](configuration) - it downloads and scores messages exactly as
described above, but stops before moving anything.

## Learning from your corrections
`mailsort` does not try to be clever about disagreements - it relies entirely on your folders. If you move an email
to a different folder than the one `mailsort` suggested, that correction becomes part of the training data the next
time models are retrained (step 3), because training always reads the folder an email currently sits in, not the
folder `mailsort` last predicted. There is no separate feedback API to call; simply sorting your email as you
normally would is enough.

## What is stored, and where
The local database contains:

* your email metadata (subject, thread, sender, recipients, folder and dates) - not a shared, cross-user index. Each
  row is scoped to a `db_user_id`, so a single database can serve multiple accounts without mixing their data.
* your trained models, serialized with [pickle](https://docs.python.org/3/library/pickle.html).

This database lives exclusively wherever you point the `-d/--database` connection string - typically a local SQLite
file on your own machine. Nothing is shared with any third party.

## Limitations to be aware of
* The model only learns from folders you already filed emails into by hand, so a brand-new folder with very few
  emails behind it will rarely reach the 90% confidence needed to be suggested - this is intentional, to avoid
  confidently wrong guesses, but it does mean new folders take a little time to "warm up".
* Because the model deliberately ignores email body text, two emails with very similar content but no overlapping
  sender, recipients or thread will not be linked by `mailsort` today. See
  [Developer - Future directions](developer) for the direction this may take next.
