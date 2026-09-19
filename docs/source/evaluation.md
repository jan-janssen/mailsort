# Evaluation and confidence
`mailsort` decides whether to move a message automatically by comparing a machine learning score against
`recommendation_ratio` (90% by default). It is tempting to read `recommendation_ratio=0.9` as "90% certainty", but
that is not automatically true, and earlier versions of this documentation did not make that distinction. This page
explains what the score actually is, what `mailsort` does about it, and how to check whether a threshold is
trustworthy for your own mailbox before you let it move mail automatically.

## Why a classifier score is not automatically a probability
`mailsort` trains one [random forest](https://en.wikipedia.org/wiki/Random_forest) classifier per folder. A random
forest's score for "this email belongs to folder X" is the fraction of trees in the forest that voted for X - a
genuinely useful signal for ranking folders against each other, but not, on its own, a calibrated probability.
Concretely: among every email a forest ever scores at 0.9, the fraction that actually belong to that folder is not
guaranteed to be 90%. Tree ensembles are well known to produce scores pulled toward the middle of the 0-1 range
(Niculescu-Mizil & Caruana, 2005, *"Predicting Good Probabilities With Supervised Learning"*), which is exactly the
kind of systematic bias that makes "90% score" and "90% correct" different claims.

## Calibration - and why it is conditional
`mailsort` addresses this with [Platt scaling](https://en.wikipedia.org/wiki/Platt_scaling) (scikit-learn's
`CalibratedClassifierCV` with `method="sigmoid"`), which rescales a classifier's raw scores against cross-validated
held-out folds of its own training data so that, empirically, X% of messages scored X actually do belong to the
predicted folder. Two choices behind this deserve to be explicit:

- **Sigmoid, not isotonic.** scikit-learn also offers isotonic regression, which is more flexible but needs
  substantially more data to avoid overfitting - the scikit-learn user guide, echoing the same 2005 paper, suggests
  it only pays off with on the order of 1000+ calibration samples. Individual IMAP folders rarely have anywhere
  near that many messages, so `mailsort` always uses the more data-efficient sigmoid fit.
- **Calibration is per folder, and conditional on having enough data.** A folder's classifier is only calibrated
  if both classes ("belongs here" / "does not") have at least 20 training examples each (configurable via
  `min_samples_per_class_for_calibration`) - roughly enough for each cross-validation fold to see a handful of
  positive examples. Below that floor, calibrating would fit the rescaling function on noise, which is worse than
  not calibrating at all. Folders that do not meet the floor keep their raw, uncalibrated score instead - `mailsort`
  never presents an unreliable calibration as if it were a reliable one.

Every [`Prediction`](developer) - what `mailsort predict`/`sort`/`MailSorter.predict()`/`sort()` produce - carries a
`score_type` field (`"calibrated"` or `"raw"`) so this distinction is explicit wherever a score is used: in the CLI
table, in a caller's own code (such as [gmailsorter](https://github.com/jan-janssen/gmailsorter)), in a future web
interface, or in an audit log. **Calibration also does not make the classifier more accurate** - it makes the score
honest about the classifier's actual accuracy, which may still be poor. Check both, using the tool below.

## `mailsort evaluate`
To check whether a `recommendation_ratio` is trustworthy for your own mailbox - rather than assuming it is - run:
```
mailsort evaluate
```
This trains a **separate, throwaway** set of per-folder models on part of your local database and scores them
against the rest, then reports:
- **Precision, recall, F1 and support**, per folder, at `recommendation_ratio` (90% by default,
  `--recommendation-ratio` to change it).
- **Coverage** - the fraction of held-out messages the threshold would act on automatically - and the
  **abstention rate** (`1 - coverage`).
- **Confusion information** - which folders get confused with which, among messages the system was confident
  enough to act on.
- Whether each folder's classifier ended up calibrated or raw (see above).

It never touches the models `mailsort train` has already stored - running it does not change what `mailsort sort`
would do next; it only estimates how well that training procedure is likely to generalize.

### Reading the report: precision over recall
Automatically moving a message to the wrong folder is more costly than leaving it unsorted: a wrong move can hide a
message from view and has to be found and corrected by hand later, while an unsorted message just needs a moment's
manual filing, same as before `mailsort` was involved. The report's `Overall precision among accepted` line is the
number to weigh most heavily for that reason - it answers "if I trust this system to act at this threshold, how
often will it be right", using only the messages the system was confident enough to act on. Recall is diluted by
cheap abstentions, not just expensive wrong moves, so a low recall with a high precision is a perfectly reasonable
place to operate; a low precision at any recall is not.

When choosing a threshold, compare a few of them rather than assuming the default fits your data:
```
mailsort evaluate --sweep
```
This reports coverage and precision at several thresholds (0.5, 0.7, 0.8, 0.9, 0.95, 0.99) from a single trained
split, so you can pick the highest threshold whose coverage you can live with, rather than the lowest one whose
precision merely looks acceptable.

### Avoiding data leakage
`mailsort`'s feature encoding includes each message's email thread as a feature. If two messages from the same
thread ended up on opposite sides of a naive train/test split, a classifier could trivially "recognise" the thread
it was trained on, rather than generalising from sender/recipient patterns - silently inflating every metric above.
`mailsort evaluate` avoids this by splitting whole email threads, never individual messages, between the training
and test portions - no thread's messages are ever split across both sides.

### Limitations
- This is a held-out evaluation of a *freshly trained* set of models on *your* data at the time you ran it, not a
  universal accuracy figure - it will drift as your mailbox and folders change, so re-run it periodically,
  especially after adding or renaming folders.
- Small test sets produce noisy metrics. The report includes support counts (and train/test message counts)
  explicitly so you can judge how much to trust a given folder's precision/recall - a folder evaluated on 3
  messages should be trusted far less than one evaluated on 80.
- A message whose folder was never seen during training - for example a very small or brand-new folder that landed
  entirely in the held-out test portion by chance - cannot be recommended for at all. Such messages are excluded
  from the report rather than counted as a miss the model had no way to avoid; the excluded count is reported too.
- `min_samples_per_class_for_calibration` (20 by default) is a pragmatic, configurable floor informed by how many
  examples a cross-validation fold needs to be stable, not a number derived from your data specifically. Use
  `mailsort evaluate` itself, with `--no-calibration` to compare, rather than trusting the default blindly.

See [Configuration](configuration) for the full `mailsort evaluate`/`mailsort train --no-calibration` command line
reference, and [Developer](developer) for the equivalent Python API
(`mailsort.evaluation`/`mailsort.ml.calibration`).
