import random
from datetime import datetime, timedelta
from unittest import TestCase

import pandas as pd

from mailsort.ml.evaluation import (
    EvaluationReport,
    HoldoutScores,
    ScoredTestMessage,
    evaluate_at_threshold,
    evaluate_machine_learning_models,
    evaluate_thresholds,
    score_holdout_split,
    thread_grouped_train_test_split,
)


def _scored(
    message_id,
    true_folder_lst,
    recommended_folder,
    score,
    calibrated=False,
):
    return ScoredTestMessage(
        message_id=message_id,
        true_folder_lst=true_folder_lst,
        recommended_folder=recommended_folder,
        score=score,
        calibrated=calibrated,
    )


def _holdout(
    scored_message_lst, trained_label_lst=None, calibration_status=None, **kwargs
):
    if trained_label_lst is None:
        trained_label_lst = sorted(
            {
                folder
                for scored in scored_message_lst
                for folder in scored.true_folder_lst
            }
        )
    if calibration_status is None:
        calibration_status = dict.fromkeys(trained_label_lst, False)
    defaults = {
        "train_message_count": 100,
        "test_message_count": len(scored_message_lst),
        "excluded_test_message_count": 0,
        "trained_label_lst": trained_label_lst,
        "calibration_status": calibration_status,
        "scored_message_lst": scored_message_lst,
    }
    defaults.update(kwargs)
    return HoldoutScores(**defaults)


class ThreadGroupedTrainTestSplitTest(TestCase):
    def _make_df(self, message_count, thread_size):
        return pd.DataFrame(
            [
                {
                    "id": f"id{i}",
                    "threads": f"thread{i // thread_size}",
                    "labels": ["Inbox"],
                    "from": "a@b.com",
                    "to": [],
                    "cc": [],
                    "subject": "s",
                    "content": None,
                    "date": datetime(2024, 1, 1) + timedelta(hours=i),
                }
                for i in range(message_count)
            ]
        )

    def test_no_thread_is_split_across_train_and_test(self):
        df = self._make_df(message_count=100, thread_size=4)

        df_train, df_test = thread_grouped_train_test_split(
            df, test_size=0.3, random_state=1
        )

        train_threads = set(df_train["threads"])
        test_threads = set(df_test["threads"])
        self.assertEqual(train_threads & test_threads, set())
        self.assertEqual(len(df_train) + len(df_test), len(df))

    def test_reproducible_with_the_same_random_state(self):
        df = self._make_df(message_count=60, thread_size=3)

        df_train_a, df_test_a = thread_grouped_train_test_split(
            df, test_size=0.25, random_state=7
        )
        df_train_b, df_test_b = thread_grouped_train_test_split(
            df, test_size=0.25, random_state=7
        )

        self.assertEqual(sorted(df_train_a["id"]), sorted(df_train_b["id"]))
        self.assertEqual(sorted(df_test_a["id"]), sorted(df_test_b["id"]))

    def test_different_random_state_can_produce_a_different_split(self):
        df = self._make_df(message_count=200, thread_size=2)

        _, df_test_a = thread_grouped_train_test_split(
            df, test_size=0.3, random_state=1
        )
        _, df_test_b = thread_grouped_train_test_split(
            df, test_size=0.3, random_state=2
        )

        self.assertNotEqual(sorted(df_test_a["id"]), sorted(df_test_b["id"]))


class EvaluateAtThresholdTest(TestCase):
    def test_accepted_prediction_is_a_true_positive(self):
        holdout = _holdout(
            [_scored("id1", ["Inbox"], "Inbox", score=0.95, calibrated=True)]
        )

        report = evaluate_at_threshold(holdout, recommendation_ratio=0.9)

        self.assertEqual(report.accepted_count, 1)
        self.assertEqual(report.abstained_count, 0)
        self.assertEqual(report.coverage, 1.0)
        self.assertEqual(report.accepted_correct_count, 1)
        self.assertEqual(report.overall_precision, 1.0)
        (fm,) = report.folder_metrics
        self.assertEqual(fm.folder, "Inbox")
        self.assertEqual((fm.support, fm.true_positive, fm.false_positive), (1, 1, 0))
        self.assertEqual((fm.misrouted, fm.abstained), (0, 0))
        self.assertEqual((fm.precision, fm.recall, fm.f1), (1.0, 1.0, 1.0))
        self.assertEqual(report.confusion, {"Inbox": {"Inbox": 1}})

    def test_low_score_abstains_and_is_not_a_penalized_miss(self):
        # The model has an opinion (recommended_folder is set) but is not confident enough - this
        # must show up as an abstention, not as a false negative that looks like the model was
        # simply wrong.
        holdout = _holdout([_scored("id1", ["Inbox"], "Inbox", score=0.4)])

        report = evaluate_at_threshold(holdout, recommendation_ratio=0.9)

        self.assertEqual(report.accepted_count, 0)
        self.assertEqual(report.abstained_count, 1)
        self.assertEqual(report.coverage, 0.0)
        self.assertIsNone(report.overall_precision)
        (fm,) = report.folder_metrics
        self.assertEqual(fm.abstained, 1)
        self.assertEqual((fm.true_positive, fm.false_positive, fm.misrouted), (0, 0, 0))
        self.assertIsNone(fm.precision)
        self.assertEqual(fm.recall, 0.0)
        self.assertIsNone(fm.f1)
        self.assertEqual(report.confusion, {})

    def test_score_exactly_at_threshold_is_not_accepted(self):
        # Strict ">" cutoff, matching mailsort.ml.model.score_messages_with_machine_learning_models
        # and AbstractMailBox.get_label_recommendations() - a tie must not be treated as a win.
        holdout = _holdout([_scored("id1", ["Inbox"], "Inbox", score=0.9)])

        report = evaluate_at_threshold(holdout, recommendation_ratio=0.9)

        self.assertEqual(report.accepted_count, 0)
        self.assertEqual(report.abstained_count, 1)

    def test_confident_wrong_move_is_a_false_positive_and_misroute(self):
        holdout = _holdout(
            [_scored("id1", ["Inbox"], "Spam", score=0.95, calibrated=True)]
        )

        report = evaluate_at_threshold(holdout, recommendation_ratio=0.9)

        self.assertEqual(report.accepted_count, 1)
        self.assertEqual(report.accepted_correct_count, 0)
        self.assertEqual(report.overall_precision, 0.0)
        by_folder = {fm.folder: fm for fm in report.folder_metrics}
        self.assertEqual(by_folder["Inbox"].misrouted, 1)
        self.assertEqual(by_folder["Inbox"].recall, 0.0)
        self.assertEqual(by_folder["Spam"].false_positive, 1)
        # Spam was recommended, always wrongly (it has no true support in this test) - precision
        # is well-defined at 0.0, not None; None is reserved for "never recommended at all".
        self.assertEqual(by_folder["Spam"].precision, 0.0)
        self.assertEqual(report.confusion, {"Inbox": {"Spam": 1}})

    def test_mixed_batch_precision_recall_and_coverage(self):
        holdout = _holdout(
            [
                _scored("id1", ["Inbox"], "Inbox", score=0.95, calibrated=True),
                _scored("id2", ["Inbox"], "Inbox", score=0.95, calibrated=True),
                _scored("id3", ["Inbox"], "Spam", score=0.95, calibrated=True),
                _scored("id4", ["Inbox"], "Inbox", score=0.4),
                _scored("id5", ["Spam"], "Spam", score=0.95, calibrated=True),
            ]
        )

        report = evaluate_at_threshold(holdout, recommendation_ratio=0.9)

        self.assertEqual(report.test_message_count, 5)
        self.assertEqual(report.accepted_count, 4)
        self.assertEqual(report.abstained_count, 1)
        self.assertEqual(report.coverage, 4 / 5)
        self.assertEqual(report.accepted_correct_count, 3)
        self.assertEqual(report.overall_precision, 3 / 4)

        by_folder = {fm.folder: fm for fm in report.folder_metrics}
        inbox = by_folder["Inbox"]
        self.assertEqual(inbox.support, 4)
        self.assertEqual(inbox.true_positive, 2)
        self.assertEqual(inbox.misrouted, 1)
        self.assertEqual(inbox.abstained, 1)
        self.assertEqual(inbox.recall, 0.5)
        spam = by_folder["Spam"]
        self.assertEqual(spam.support, 1)
        self.assertEqual(spam.true_positive, 1)
        self.assertEqual(spam.false_positive, 1)
        self.assertEqual(spam.precision, 0.5)
        self.assertEqual(spam.recall, 1.0)

    def test_multi_label_true_folders_count_as_a_hit_for_either(self):
        # A message truly filed in two folders at once is a rare edge case, but the true_folder
        # set is checked with "in", so recommending either one is treated as correct.
        holdout = _holdout(
            [_scored("id1", ["Inbox", "Important"], "Important", score=0.95)]
        )

        report = evaluate_at_threshold(holdout, recommendation_ratio=0.9)

        self.assertEqual(report.accepted_correct_count, 1)
        by_folder = {fm.folder: fm for fm in report.folder_metrics}
        self.assertEqual(by_folder["Important"].true_positive, 1)
        # Inbox has support (it was one of the true folders) but was not the one recommended -
        # counted as a miss for Inbox specifically, even though the message was correctly caught.
        self.assertEqual(by_folder["Inbox"].misrouted, 1)

    def test_empty_holdout_produces_a_report_without_crashing(self):
        holdout = _holdout([], trained_label_lst=["Inbox"])

        report = evaluate_at_threshold(holdout, recommendation_ratio=0.9)

        self.assertEqual(report.test_message_count, 0)
        self.assertEqual(report.accepted_count, 0)
        self.assertEqual(report.coverage, 0.0)
        self.assertIsNone(report.overall_precision)
        self.assertEqual(report.folder_metrics, [])
        self.assertEqual(report.confusion, {})

    def test_excluded_message_count_is_carried_through_unchanged(self):
        holdout = _holdout(
            [_scored("id1", ["Inbox"], "Inbox", score=0.95)],
            excluded_test_message_count=3,
        )

        report = evaluate_at_threshold(holdout, recommendation_ratio=0.9)

        self.assertEqual(report.excluded_message_count, 3)

    def test_calibration_status_is_carried_through_unchanged(self):
        holdout = _holdout(
            [_scored("id1", ["Inbox"], "Inbox", score=0.95, calibrated=True)],
            calibration_status={"Inbox": True, "Spam": False},
        )

        report = evaluate_at_threshold(holdout, recommendation_ratio=0.9)

        self.assertEqual(report.calibration_status, {"Inbox": True, "Spam": False})


class EvaluateThresholdsTest(TestCase):
    def test_returns_one_report_per_threshold_in_order(self):
        holdout = _holdout(
            [
                _scored("id1", ["Inbox"], "Inbox", score=0.6),
                _scored("id2", ["Inbox"], "Inbox", score=0.95),
            ]
        )

        reports = evaluate_thresholds(holdout, [0.5, 0.9, 0.99])

        self.assertEqual([r.recommendation_ratio for r in reports], [0.5, 0.9, 0.99])
        self.assertEqual([r.accepted_count for r in reports], [2, 1, 0])

    def test_does_not_retrain_between_thresholds(self):
        # evaluate_thresholds() must be a pure function of the already-scored holdout - the same
        # ScoredTestMessage list is reused, not rebuilt, across every threshold.
        holdout = _holdout([_scored("id1", ["Inbox"], "Inbox", score=0.8)])

        evaluate_thresholds(holdout, [0.1, 0.5, 0.9])

        self.assertEqual(len(holdout.scored_message_lst), 1)


def _synthetic_email_df(row_count=300, folder_lst=("Inbox", "Receipts", "Newsletters")):
    """
    A synthetic, well-separated dataset: each folder has its own small pool of senders, so a
    sender-based classifier should learn it almost perfectly - useful for exercising the full
    score_holdout_split()/evaluate_machine_learning_models() pipeline end to end without relying
    on real mail.
    """
    rng = random.Random(0)
    senders_by_folder = {
        folder: [f"{folder.lower()}{n}@example.com" for n in range(2)]
        for folder in folder_lst
    }
    rows = []
    for i in range(row_count):
        folder = folder_lst[i % len(folder_lst)]
        rows.append(
            {
                "id": f"id{i}",
                "from": rng.choice(senders_by_folder[folder]),
                "to": ["me@example.com"],
                "cc": [],
                "date": datetime(2024, 1, 1) + timedelta(hours=i),
                "threads": f"thread{i // 2}",
                "labels": [folder],
                "subject": f"Subject {i}",
                "content": None,
            }
        )
    return pd.DataFrame(rows)


class ScoreHoldoutSplitIntegrationTest(TestCase):
    """
    Integration tests that exercise score_holdout_split()/evaluate_machine_learning_models()
    end to end (real feature encoding, real RandomForestClassifier training) - kept fast with a
    small n_estimators and max_workers=1, matching the convention already used in test_ml.py.
    """

    def test_empty_dataframe_does_not_crash(self):
        df = pd.DataFrame(
            columns=[
                "id",
                "from",
                "to",
                "cc",
                "date",
                "threads",
                "labels",
                "subject",
                "content",
            ]
        )

        holdout = score_holdout_split(df, test_size=0.3, random_state=0, max_workers=1)

        self.assertEqual(holdout.train_message_count, 0)
        self.assertEqual(holdout.test_message_count, 0)
        self.assertEqual(holdout.trained_label_lst, [])

        report = evaluate_at_threshold(holdout, recommendation_ratio=0.9)
        self.assertEqual(report.test_message_count, 0)
        self.assertIsNone(report.overall_precision)

    def test_well_separated_data_reaches_high_coverage_and_precision(self):
        df = _synthetic_email_df(row_count=240)

        holdout = score_holdout_split(
            df, test_size=0.3, random_state=0, n_estimators=30, max_workers=1
        )
        report = evaluate_at_threshold(holdout, recommendation_ratio=0.9)

        self.assertGreater(holdout.test_message_count, 0)
        self.assertEqual(holdout.excluded_test_message_count, 0)
        self.assertGreater(report.coverage, 0.8)
        self.assertGreater(report.overall_precision, 0.9)

    def test_small_dataset_falls_back_to_raw_uncalibrated_scores(self):
        df = _synthetic_email_df(row_count=16, folder_lst=("Inbox", "Spam"))

        holdout = score_holdout_split(
            df, test_size=0.3, random_state=0, n_estimators=10, max_workers=1
        )

        self.assertTrue(holdout.trained_label_lst)
        self.assertFalse(any(holdout.calibration_status.values()))

    def test_large_dataset_calibrates(self):
        df = _synthetic_email_df(row_count=240, folder_lst=("Inbox", "Spam"))

        holdout = score_holdout_split(
            df, test_size=0.3, random_state=0, n_estimators=30, max_workers=1
        )

        self.assertTrue(any(holdout.calibration_status.values()))

    def test_no_calibration_flag_disables_it_even_with_plenty_of_data(self):
        df = _synthetic_email_df(row_count=240, folder_lst=("Inbox", "Spam"))

        holdout = score_holdout_split(
            df,
            test_size=0.3,
            random_state=0,
            n_estimators=30,
            max_workers=1,
            calibrate=False,
        )

        self.assertFalse(any(holdout.calibration_status.values()))

    def test_folder_only_present_in_test_split_is_excluded_not_penalized(self):
        # A folder with exactly one thread can, by chance, land entirely in the test split -
        # the model never saw it during training, so it cannot possibly be recommended. Such
        # messages must be excluded rather than counted as misses the model had no way to avoid.
        rows = _synthetic_email_df(row_count=40, folder_lst=("Inbox",)).to_dict(
            "records"
        )
        rows.append(
            {
                "id": "rare1",
                "from": "rare@example.com",
                "to": [],
                "cc": [],
                "date": datetime(2024, 1, 1),
                "threads": "rare_thread",
                "labels": ["Rare"],
                "subject": "s",
                "content": None,
            }
        )
        df = pd.DataFrame(rows)

        # random_state chosen so the single-message "rare_thread" group lands in the test split
        holdout = score_holdout_split(
            df, test_size=0.5, random_state=7, n_estimators=10, max_workers=1
        )

        self.assertNotIn("Rare", holdout.trained_label_lst)
        self.assertGreaterEqual(holdout.excluded_test_message_count, 1)
        self.assertTrue(
            all(
                "Rare" not in scored.true_folder_lst
                for scored in holdout.scored_message_lst
            )
        )

    def test_evaluate_machine_learning_models_is_a_thin_wrapper(self):
        df = _synthetic_email_df(row_count=120)

        report = evaluate_machine_learning_models(
            df,
            recommendation_ratio=0.8,
            test_size=0.3,
            random_state=0,
            n_estimators=20,
            max_workers=1,
        )

        self.assertIsInstance(report, EvaluationReport)
        self.assertEqual(report.recommendation_ratio, 0.8)


if __name__ == "__main__":
    import unittest

    unittest.main()
