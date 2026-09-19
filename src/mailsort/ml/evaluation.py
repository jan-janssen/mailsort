"""
Evaluation for the per-folder machine learning models: precision, recall, F1, support, confusion
information, and coverage/abstention rate at a chosen ``recommendation_ratio``. This is the
evidence a ``recommendation_ratio`` choice should be based on, rather than treating
``recommendation_ratio`` itself as if it already were a calibrated error-rate guarantee - see
:mod:`mailsort.ml.calibration` for why a raw classifier score is not automatically a probability,
even after calibration is applied.

Evaluation never touches the models mailsort has already trained and stored (see
:meth:`mailsort.base.mail.AbstractMailBox.fit_machine_learning_model_to_database`) - it trains a
separate, throwaway set of per-folder models on a held-out training split purely to estimate how
that same training procedure performs on messages it has not seen. Running an evaluation does not
change what ``mailsort train``/``mailsort sort`` would actually do next.

Avoiding data leakage
----------------------
mailsort's feature encoding (see :mod:`mailsort.ml.encoding`) includes each message's email
thread as a categorical (one-hot) feature. If two messages from the same thread ended up on
opposite sides of a train/test split, a classifier could trivially "recognise" the thread it was
trained on rather than generalising from sender/recipient/thread patterns, silently inflating
every metric below - the thread ID becomes close to a unique row identifier once it has been seen
during training. To avoid this, :func:`thread_grouped_train_test_split` keeps every message of the
same thread on one side of the split; no thread's messages are ever split across train and test.

The operational objective: precision over recall
---------------------------------------------------
Automatically moving a message to the wrong folder is more costly than leaving it unsorted - it
can hide a message from view, and the fix has to be a manual, after-the-fact correction. Leaving
a message unsorted (the model abstaining) costs the user a moment's manual filing, once. This
evaluation reports two precision figures for exactly that reason: :attr:`EvaluationReport.
overall_precision`, precision *among accepted predictions only*, which directly answers "if I
trust this system to act at this threshold, how often will it be right" - as opposed to recall,
which is diluted by abstentions that are cheap, not by wrong moves that are expensive. When
choosing a `recommendation_ratio`, prefer the highest threshold whose coverage you can live with
over the lowest one whose precision looks acceptable.

Limitations
-----------
- This is a held-out evaluation of a *freshly trained* set of models on *your* data at the time
  you ran it - it estimates what ``mailsort train`` would produce, not a universal accuracy
  figure, and it will drift as your mailbox and folders change. Re-run it periodically, especially
  after adding or renaming folders.
- Small test sets produce noisy metrics - support counts are reported explicitly so you can judge
  how much to trust a given folder's precision/recall (a folder evaluated on 3 test messages
  should be trusted far less than one evaluated on 80).
- A message whose true folder(s) were never seen during training (e.g. a very small or brand-new
  folder that landed entirely in the test split by chance) cannot be recommended for - it is
  reported in ``excluded_message_count`` and not scored either way, rather than being silently
  counted as a miss it had no way to avoid.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas
from sklearn.model_selection import GroupShuffleSplit

from mailsort.ml.calibration import is_calibrated
from mailsort.ml.encoding import encode_df_for_machine_learning
from mailsort.ml.model import (
    fit_machine_learning_models,
    score_messages_with_machine_learning_models,
)


def thread_grouped_train_test_split(
    df: pandas.DataFrame, test_size: float = 0.25, random_state: int = 42
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    Split `df` into a train and test set, keeping every message of the same email thread
    (``df["threads"]``) entirely on one side - see the module docstring for why this matters.

    Args:
        df: emails, as returned by AbstractMailBox.get_all_emails_in_database() - must have a
            "threads" column
        test_size: fraction of *threads* (not messages) held out for testing (0<test_size<1)
        random_state: random seed, for a reproducible split

    Returns:
        (df_train, df_test)
    """
    if len(df) == 0:
        # GroupShuffleSplit requires at least one sample - an empty database is a valid, if
        # useless, input (e.g. before the first "mailsort sync"), so this returns two empty
        # frames rather than letting the caller hit a scikit-learn validation error.
        return df.reset_index(drop=True), df.reset_index(drop=True).copy()
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(df, groups=df["threads"]))
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


@dataclass(frozen=True)
class ScoredTestMessage:
    """
    One held-out test message's true folder(s) and the model's prediction for it, before any
    ``recommendation_ratio`` has been applied - the raw material :func:`evaluate_at_threshold`
    turns into metrics.

    Attributes:
        message_id: the message's id
        true_folder_lst: the message's actual folder(s) that were also part of the trained label
            set - folders the model had no way to learn are excluded here, see
            HoldoutScores.excluded_test_message_count
        recommended_folder: the folder the model scores highest for this message
        score: that folder's score (see mailsort.ml.calibration - calibrated or raw depending on
            `calibrated`)
        calibrated: whether `score` is a calibrated probability
    """

    message_id: str
    true_folder_lst: list[str]
    recommended_folder: str | None
    score: float
    calibrated: bool


@dataclass(frozen=True)
class HoldoutScores:
    """
    Result of :func:`score_holdout_split`: a trained-and-scored held-out test set, before any
    `recommendation_ratio` has been applied. Pass this to :func:`evaluate_at_threshold` to get
    metrics - cheaply, at any threshold, without retraining, so sweeping several thresholds (see
    :func:`evaluate_thresholds`) costs one training+scoring pass, not one per threshold.

    Attributes:
        train_message_count: messages used to train the held-out models
        test_message_count: held-out messages that were actually scored (had at least one true
            folder in common with the trained label set)
        excluded_test_message_count: held-out messages skipped because none of their folder(s)
            were part of the trained label set - see the module docstring
        trained_label_lst: folders a held-out model was trained for
        calibration_status: whether each trained folder's classifier ended up calibrated - see
            mailsort.ml.calibration.should_calibrate()
        scored_message_lst: per-message raw scores, see ScoredTestMessage
    """

    train_message_count: int
    test_message_count: int
    excluded_test_message_count: int
    trained_label_lst: list[str]
    calibration_status: dict[str, bool]
    scored_message_lst: list[ScoredTestMessage]


def score_holdout_split(
    df: pandas.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    n_estimators: int = 100,
    max_features: int = 400,
    bootstrap: bool = True,
    calibrate: bool = True,
    min_samples_per_class_for_calibration: int = 20,
    max_calibration_cv_folds: int = 5,
    max_workers: int | None = None,
    label_prefix: str = "labels_",
) -> HoldoutScores:
    """
    Split `df` into a thread-grouped train/test set (see thread_grouped_train_test_split), train
    a throwaway set of per-folder models on the train split exactly as
    AbstractMailBox.fit_machine_learning_model_to_database() would, and score the test split
    exactly as AbstractMailBox.get_label_recommendations() would - the raw material
    evaluate_at_threshold() turns into metrics.

    Args:
        df: emails, as returned by AbstractMailBox.get_all_emails_in_database()
        test_size: fraction of threads held out for testing (0<test_size<1)
        random_state: random seed for both the split and the classifiers, for reproducibility
        n_estimators: number of estimators per classifier
        max_features: maximum number of features per classifier
        bootstrap: whether bootstrap samples are used when building trees
        calibrate: calibrate each folder's classifier when there is enough data to do so safely -
            default: True. See mailsort.ml.calibration.
        min_samples_per_class_for_calibration: see mailsort.ml.calibration.should_calibrate()
        max_calibration_cv_folds: see mailsort.ml.calibration.calibration_cv_folds()
        max_workers: maximum number of worker processes to train models in parallel
        label_prefix: prefix used to recognise label columns during feature encoding

    Returns:
        HoldoutScores
    """
    df_train, df_test = thread_grouped_train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    if len(df_train) == 0:
        # Nothing to train on (e.g. an empty database) - encode_df_for_machine_learning() does
        # not handle a fully empty input, so this is short-circuited rather than hitting that
        # unrelated edge case here.
        return HoldoutScores(
            train_message_count=0,
            test_message_count=0,
            excluded_test_message_count=len(df_test),
            trained_label_lst=[],
            calibration_status={},
            scored_message_lst=[],
        )

    # Mirrors AbstractMailBox.fit_machine_learning_model_to_database() - deduplicating and
    # sorting columns is duplicated here rather than shared, to keep that method's existing,
    # separately-tested feature-preparation call sites stable; see mailsort/training.py for the
    # same trade-off made previously.
    df_train_features, df_train_labels = encode_df_for_machine_learning(
        df=df_train,
        feature_lst=[],
        label_lst=[],
        return_labels=True,
        label_prefix=label_prefix,
    )
    df_train_features = df_train_features.loc[
        :, ~df_train_features.columns.duplicated()
    ].copy()
    df_train_features = df_train_features.reindex(
        sorted(df_train_features.columns), axis=1
    )
    model_dict = fit_machine_learning_models(
        df_all_features=df_train_features,
        df_all_labels=df_train_labels,
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=random_state,
        bootstrap=bootstrap,
        max_workers=max_workers,
        calibrate=calibrate,
        min_samples_per_class_for_calibration=min_samples_per_class_for_calibration,
        max_calibration_cv_folds=max_calibration_cv_folds,
    )
    trained_label_set = set(model_dict.keys())
    # MachineLearningDatabase.store_models() strips "email_id" from the feature list it persists
    # (see mailsort/ml/database.py) before AbstractMailBox.get_label_recommendations() reloads
    # and reuses it - mirrored here, since encode_df_for_machine_learning() appends its own
    # "email_id" column unconditionally and would otherwise end up with two.
    trained_feature_lst = [
        feature
        for feature in df_train_features.columns.tolist()
        if feature != "email_id"
    ]

    scored_message_lst: list[ScoredTestMessage] = []
    excluded_test_message_count = 0
    if len(df_test) > 0 and len(model_dict) > 0:
        # Mirrors AbstractMailBox.get_label_recommendations() - see the comment above.
        df_test_features = encode_df_for_machine_learning(
            df=df_test,
            feature_lst=trained_feature_lst,
            label_lst=list(trained_label_set),
            return_labels=False,
            label_prefix=label_prefix,
        )
        df_test_features = df_test_features.reindex(
            sorted(df_test_features.columns), axis=1
        )
        score_lst = score_messages_with_machine_learning_models(
            df_features=df_test_features,
            model_dict=model_dict,
        )
        score_by_id = {entry["email_id"]: entry for entry in score_lst}
    else:
        score_by_id = {}

    for _, row in df_test.iterrows():
        true_folder_lst = [
            folder for folder in row["labels"] if folder in trained_label_set
        ]
        if not true_folder_lst:
            excluded_test_message_count += 1
            continue
        entry = score_by_id.get(row["id"])
        if entry is None:
            scored_message_lst.append(
                ScoredTestMessage(
                    message_id=row["id"],
                    true_folder_lst=true_folder_lst,
                    recommended_folder=None,
                    score=0.0,
                    calibrated=False,
                )
            )
        else:
            scored_message_lst.append(
                ScoredTestMessage(
                    message_id=row["id"],
                    true_folder_lst=true_folder_lst,
                    recommended_folder=entry["recommended_label"],
                    score=entry["score"],
                    calibrated=entry["calibrated"],
                )
            )

    return HoldoutScores(
        train_message_count=len(df_train),
        test_message_count=len(scored_message_lst),
        excluded_test_message_count=excluded_test_message_count,
        trained_label_lst=sorted(trained_label_set),
        calibration_status={
            label: is_calibrated(model) for label, model in model_dict.items()
        },
        scored_message_lst=scored_message_lst,
    )


@dataclass(frozen=True)
class FolderMetrics:
    """
    Precision/recall/F1/support for one folder at a chosen `recommendation_ratio`.

    Precision and recall are charged to different folders for the same wrong move, by design:
    a message truly belonging to "Inbox" but wrongly moved to "Spam" counts as a recall failure
    for "Inbox" (support was there, the system did not correctly claim it) and a precision
    failure for "Spam" (the system claimed a message that was not its to claim) - both are true,
    and neither folder's number alone tells the whole story.

    Attributes:
        folder: the folder name
        support: number of held-out test messages truly belonging to this folder
        true_positive: correctly accepted and recommended to this folder
        false_positive: accepted and recommended to this folder for a message that does not
            belong here
        misrouted: this folder's true messages that were accepted, but recommended to a
            *different* folder
        abstained: this folder's true messages the model did not accept (recommend, but not
            confidently enough - or not at all)
        precision: true_positive / (true_positive + false_positive) - None if this folder was
            never recommended at all (undefined, not zero)
        recall: true_positive / support - 0.0 if support is 0
        f1: harmonic mean of precision and recall - None if precision is None
    """

    folder: str
    support: int
    true_positive: int
    false_positive: int
    misrouted: int
    abstained: int
    precision: float | None
    recall: float
    f1: float | None


@dataclass(frozen=True)
class EvaluationReport:
    """
    Precision/recall/F1/support, confusion information, and coverage/abstention rate for a held-
    out test set at one `recommendation_ratio` - see :func:`evaluate_at_threshold`.

    Attributes:
        recommendation_ratio: the threshold this report was computed at
        train_message_count: messages used to train the held-out models
        test_message_count: held-out messages that were scored (see HoldoutScores)
        excluded_message_count: held-out messages that could not be scored at all (see
            HoldoutScores.excluded_test_message_count) - constant across thresholds
        accepted_count: test messages whose score cleared `recommendation_ratio`
        abstained_count: test messages whose score did not - `test_message_count -
            accepted_count`
        coverage: `accepted_count / test_message_count` - the fraction of messages this threshold
            would act on automatically; `0.0` if there were no test messages
        accepted_correct_count: accepted predictions whose recommended folder was actually
            correct
        overall_precision: `accepted_correct_count / accepted_count` across every folder - the
            single number that most directly answers "if I trust this system at this threshold,
            how often is it right" (see the module docstring on the operational objective); None
            if nothing was accepted
        folder_metrics: per-folder precision/recall/F1/support, for folders with test support > 0
        confusion: `confusion[true_folder][recommended_folder]` = count, over *accepted*
            predictions only (abstained predictions are not a "confusion", they took no action)
        calibration_status: whether each trained folder's classifier ended up calibrated
    """

    recommendation_ratio: float
    train_message_count: int
    test_message_count: int
    excluded_message_count: int
    accepted_count: int
    abstained_count: int
    coverage: float
    accepted_correct_count: int
    overall_precision: float | None
    folder_metrics: list[FolderMetrics]
    confusion: dict[str, dict[str, int]]
    calibration_status: dict[str, bool]


def evaluate_at_threshold(
    holdout: HoldoutScores, recommendation_ratio: float = 0.9
) -> EvaluationReport:
    """
    Turn already-computed HoldoutScores into metrics at `recommendation_ratio` - cheap, since no
    retraining or rescoring happens here; call this repeatedly with different ratios (see
    evaluate_thresholds()) to compare thresholds without paying for score_holdout_split() again.

    Args:
        holdout: see score_holdout_split()
        recommendation_ratio: cutoff a score must strictly clear to be "accepted" - the same
            comparison AbstractMailBox.filter_messages_from_server() uses

    Returns:
        EvaluationReport
    """
    accepted_count = 0
    accepted_correct_count = 0
    folder_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "support": 0,
            "true_positive": 0,
            "false_positive": 0,
            "misrouted": 0,
            "abstained": 0,
        }
    )
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for scored in holdout.scored_message_lst:
        accepted = scored.score > recommendation_ratio
        if accepted:
            accepted_count += 1
            if scored.recommended_folder in scored.true_folder_lst:
                accepted_correct_count += 1
            else:
                folder_stats[scored.recommended_folder]["false_positive"] += 1
        for true_folder in scored.true_folder_lst:
            folder_stats[true_folder]["support"] += 1
            if not accepted:
                folder_stats[true_folder]["abstained"] += 1
            elif scored.recommended_folder == true_folder:
                folder_stats[true_folder]["true_positive"] += 1
                confusion[true_folder][scored.recommended_folder] += 1
            else:
                folder_stats[true_folder]["misrouted"] += 1
                confusion[true_folder][scored.recommended_folder] += 1

    folder_metrics = []
    for folder in sorted(folder_stats):
        stats = folder_stats[folder]
        denom = stats["true_positive"] + stats["false_positive"]
        precision = stats["true_positive"] / denom if denom > 0 else None
        recall = (
            stats["true_positive"] / stats["support"] if stats["support"] > 0 else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and (precision + recall) > 0
            else None
        )
        folder_metrics.append(
            FolderMetrics(
                folder=folder,
                support=stats["support"],
                true_positive=stats["true_positive"],
                false_positive=stats["false_positive"],
                misrouted=stats["misrouted"],
                abstained=stats["abstained"],
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    test_message_count = holdout.test_message_count
    return EvaluationReport(
        recommendation_ratio=recommendation_ratio,
        train_message_count=holdout.train_message_count,
        test_message_count=test_message_count,
        excluded_message_count=holdout.excluded_test_message_count,
        accepted_count=accepted_count,
        abstained_count=test_message_count - accepted_count,
        coverage=accepted_count / test_message_count if test_message_count > 0 else 0.0,
        accepted_correct_count=accepted_correct_count,
        overall_precision=(
            accepted_correct_count / accepted_count if accepted_count > 0 else None
        ),
        folder_metrics=folder_metrics,
        confusion={folder: dict(counts) for folder, counts in confusion.items()},
        calibration_status=dict(holdout.calibration_status),
    )


def evaluate_thresholds(
    holdout: HoldoutScores, recommendation_ratio_lst: list[float]
) -> list[EvaluationReport]:
    """
    evaluate_at_threshold() for several thresholds at once, from the same HoldoutScores - the
    cheap way to compare coverage/precision trade-offs (see the module docstring) without
    retraining or rescoring per threshold.

    Args:
        holdout: see score_holdout_split()
        recommendation_ratio_lst: thresholds to evaluate, e.g. [0.5, 0.7, 0.9, 0.95]

    Returns:
        list[EvaluationReport]: one report per threshold, in the same order
    """
    return [
        evaluate_at_threshold(holdout, recommendation_ratio=ratio)
        for ratio in recommendation_ratio_lst
    ]


def evaluate_machine_learning_models(
    df: pandas.DataFrame,
    recommendation_ratio: float = 0.9,
    test_size: float = 0.25,
    random_state: int = 42,
    n_estimators: int = 100,
    max_features: int = 400,
    bootstrap: bool = True,
    calibrate: bool = True,
    min_samples_per_class_for_calibration: int = 20,
    max_calibration_cv_folds: int = 5,
    max_workers: int | None = None,
    label_prefix: str = "labels_",
) -> EvaluationReport:
    """
    One-shot convenience wrapper: score_holdout_split() followed by evaluate_at_threshold() at a
    single `recommendation_ratio`. Prefer calling score_holdout_split() once yourself and
    evaluate_thresholds() to compare several ratios - each call to this function retrains from
    scratch.

    Args:
        df: emails, as returned by AbstractMailBox.get_all_emails_in_database()
        recommendation_ratio: see evaluate_at_threshold()
        (remaining args): see score_holdout_split()

    Returns:
        EvaluationReport
    """
    holdout = score_holdout_split(
        df,
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators,
        max_features=max_features,
        bootstrap=bootstrap,
        calibrate=calibrate,
        min_samples_per_class_for_calibration=min_samples_per_class_for_calibration,
        max_calibration_cv_folds=max_calibration_cv_folds,
        max_workers=max_workers,
        label_prefix=label_prefix,
    )
    return evaluate_at_threshold(holdout, recommendation_ratio=recommendation_ratio)
