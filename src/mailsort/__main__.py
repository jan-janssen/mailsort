"""
Command line interface for mailsort.

This module only parses arguments and formats output - the actual work is done by
mailsort.Imap and the reusable functions in mailsort.training / mailsort.status / mailsort.evaluation.
"""

import argparse
import sys

from mailsort import Imap
from mailsort.evaluation import (
    evaluate_at_threshold,
    evaluate_thresholds,
    score_holdout,
)
from mailsort.status import get_database_status
from mailsort.training import train_machine_learning_models

_DEFAULT_DATABASE = "sqlite:///email.db"
_DEFAULT_RECOMMENDATION_RATIO = 0.9
_DEFAULT_LABEL_PREFIX = "labels_"

_EXIT_OK = 0
_EXIT_RUNTIME_ERROR = 1
_EXIT_CONFIG_ERROR = 2

_TABLE_COLUMN_WIDTHS = {
    "message_id": 36,
    "subject": 40,
    "recommended_folder": 20,
}


def _format_recommendations_table(predictions):
    """
    Render the list[Prediction] returned by Imap.get_label_recommendations() as a plain text
    table for the command line - one row per message, no messages are moved to build this.

    Args:
        predictions (list[mailsort.results.Prediction]): return value of
            AbstractMailBox.get_label_recommendations()

    Returns:
        str: human-readable table, or a placeholder message if there are no messages
    """
    if not predictions:
        return "No messages found in this folder."
    widths = _TABLE_COLUMN_WIDTHS
    header = (
        f"{'MESSAGE ID':<{widths['message_id']}} "
        f"{'SUBJECT':<{widths['subject']}} "
        f"{'RECOMMENDED FOLDER':<{widths['recommended_folder']}} "
        f"{'SCORE':>6} {'TYPE':<10} {'ACCEPTED':>8}"
    )
    lines = [header, "-" * len(header)]
    for prediction in predictions:
        message_id = str(prediction.message_id)[: widths["message_id"]]
        subject = str(prediction.subject or "")[: widths["subject"]]
        recommended_folder = str(prediction.recommended_folder or "-")[
            : widths["recommended_folder"]
        ]
        lines.append(
            f"{message_id:<{widths['message_id']}} "
            f"{subject:<{widths['subject']}} "
            f"{recommended_folder:<{widths['recommended_folder']}} "
            f"{prediction.score:>6.2f} {prediction.score_type.value:<10} "
            f"{str(prediction.accepted):>8}"
        )
    return "\n".join(lines)


def _format_status_table(status):
    """
    Render a mailsort.status.DatabaseStatus as human-readable text for the command line.
    """
    trained_labels = (
        ", ".join(status.trained_label_lst) if status.trained_label_lst else "none"
    )
    return "\n".join(
        [
            f"Database:               {status.connection_str}",
            f"Database user id:       {status.db_user_id}",
            f"Known messages:         {status.message_count} total "
            f"({status.active_message_count} active, {status.deleted_message_count} deleted)",
            f"Trained folder models:  {len(status.trained_label_lst)} ({trained_labels})",
            f"Machine learning features stored: {status.feature_count}",
        ]
    )


def _format_ratio(value):
    return "n/a" if value is None else f"{value:.2f}"


def _format_evaluation_report(report):
    """
    Render a mailsort.ml.evaluation.EvaluationReport as human-readable text for the command line.
    """
    lines = [
        f"Evaluation at recommendation-ratio={report.recommendation_ratio:.2f} "
        f"(held-out, thread-grouped split - see 'mailsort evaluate --help')",
        f"Train messages: {report.train_message_count}   "
        f"Test messages: {report.test_message_count}   "
        f"Excluded (no trained folder in common): {report.excluded_message_count}",
        f"Coverage: {report.accepted_count}/{report.test_message_count} accepted "
        f"({report.coverage:.1%})   "
        f"Overall precision among accepted: {report.accepted_correct_count}/"
        f"{report.accepted_count} ({_format_ratio(report.overall_precision)})",
        "",
    ]
    widths = {"folder": 20, "num": 8}
    header = (
        f"{'FOLDER':<{widths['folder']}} {'SUPPORT':>{widths['num']}} "
        f"{'PRECISION':>{widths['num']}} {'RECALL':>{widths['num']}} {'F1':>{widths['num']}} "
        f"{'TP':>{widths['num']}} {'FP':>{widths['num']}} {'MISROUTED':>{widths['num']}} "
        f"{'ABSTAINED':>{widths['num']}} {'MODEL':>{widths['num']}}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for fm in report.folder_metrics:
        model = "calibrated" if report.calibration_status.get(fm.folder) else "raw"
        lines.append(
            f"{fm.folder:<{widths['folder']}} {fm.support:>{widths['num']}} "
            f"{_format_ratio(fm.precision):>{widths['num']}} "
            f"{fm.recall:>{widths['num']}.2f} {_format_ratio(fm.f1):>{widths['num']}} "
            f"{fm.true_positive:>{widths['num']}} {fm.false_positive:>{widths['num']}} "
            f"{fm.misrouted:>{widths['num']}} {fm.abstained:>{widths['num']}} "
            f"{model:>{widths['num']}}"
        )
    if report.confusion:
        lines.append("")
        lines.append("Confusion (true folder -> recommended folder, accepted only):")
        for true_folder in sorted(report.confusion):
            for recommended_folder, count in sorted(
                report.confusion[true_folder].items()
            ):
                lines.append(f"  {true_folder} -> {recommended_folder}: {count}")
    return "\n".join(lines)


def _format_evaluation_sweep(reports):
    """
    Render several mailsort.ml.evaluation.EvaluationReport instances (see
    mailsort.ml.evaluation.evaluate_thresholds) as a single comparison table.
    """
    first = reports[0]
    lines = [
        f"Coverage/precision by recommendation-ratio "
        f"(train: {first.train_message_count}, test: {first.test_message_count}, "
        f"excluded: {first.excluded_message_count})",
        f"{'RATIO':>6} {'COVERAGE':>9} {'ACCEPTED':>9} {'OVERALL PRECISION':>18}",
    ]
    for report in reports:
        lines.append(
            f"{report.recommendation_ratio:>6.2f} {report.coverage:>9.1%} "
            f"{report.accepted_count:>9} "
            f"{_format_ratio(report.overall_precision):>18}"
        )
    return "\n".join(lines)


def _add_connection_arguments(parser, suppress_defaults=False):
    # suppress_defaults=True is used for the per-subcommand copies of these arguments:
    # argparse subparsers parse into a fresh namespace and then unconditionally overwrite
    # the parent namespace's attributes with it (including untouched defaults), so without
    # SUPPRESS a global flag given before the subcommand (e.g. "mailsort --host x sync")
    # would silently be reset back to the subparser's own default.
    suppress = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument(
        "--host",
        default=suppress,
        help="IMAP server hostname e.g. imap.example.com .",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else 993,
        help="IMAP server port - default: 993 .",
    )
    parser.add_argument(
        "--username",
        default=suppress,
        help="IMAP account username.",
    )
    parser.add_argument(
        "--password",
        default=suppress,
        help="IMAP account password, e.g. an app password.",
    )
    parser.add_argument(
        "--no-ssl",
        action="store_true",
        default=argparse.SUPPRESS if suppress_defaults else False,
        help="Connect without SSL (IMAP4 instead of IMAP4_SSL).",
    )


def _add_database_argument(parser, suppress_defaults=False):
    parser.add_argument(
        "-d",
        "--database",
        default=argparse.SUPPRESS if suppress_defaults else None,
        help=(
            "Connection string to connect to the database e.g. sqlite:///email.db "
            f"- default: {_DEFAULT_DATABASE} ."
        ),
    )


def _add_identification_argument(parser, suppress_defaults=False):
    parser.add_argument(
        "-i",
        "--identification",
        default=argparse.SUPPRESS if suppress_defaults else None,
        help="User id of the database user, useful when sharing one database between "
        "multiple accounts - default: 1 .",
    )


def _add_recommendation_arguments(parser):
    parser.add_argument(
        "--recommendation-ratio",
        type=float,
        default=_DEFAULT_RECOMMENDATION_RATIO,
        help="Certainty a machine learning score must clear to count as a recommendation "
        f"(0<r<1) - default: {_DEFAULT_RECOMMENDATION_RATIO} .",
    )
    parser.add_argument(
        "--label-prefix",
        default=_DEFAULT_LABEL_PREFIX,
        help="Prefix used to recognise label columns during feature encoding - only needs "
        f"to change if you customised this in the Python API - default: {_DEFAULT_LABEL_PREFIX!r} .",
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="mailsort",
        description=(
            "Assign labels to emails on any IMAP mail server based on their similarity to "
            "other emails already assigned to the same label."
        ),
        epilog="""\
examples:
  # train the machine learning model on your already-sorted folders
  mailsort sync --host imap.example.com --username user@example.com --password "..."
  mailsort train

  # preview what sorting MailSortInbox would do, without moving anything
  mailsort predict MailSortInbox --host imap.example.com --username user@example.com --password "..."

  # actually sort MailSortInbox using the trained model
  mailsort sort MailSortInbox --host imap.example.com --username user@example.com --password "..."

  # inspect the local database
  mailsort status

  # check precision/coverage at a few thresholds before trusting one
  mailsort evaluate --sweep

Run `mailsort <command> --help` for the arguments of an individual command.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Top-level flags so they can be given before the subcommand, e.g. "mailsort --host x
    # sync" - see _add_connection_arguments for why the subcommand copies suppress their
    # defaults.
    _add_connection_arguments(parser)
    _add_database_argument(parser)
    _add_identification_argument(parser)

    subparsers = parser.add_subparsers(
        dest="command", metavar="{sync,train,predict,sort,status,evaluate}"
    )

    sync_parser = subparsers.add_parser(
        "sync",
        help="Update the local email database from the mail server.",
        description="Update the local email database from the mail server. Moves nothing "
        "and trains nothing - see 'mailsort train' to (re-)train the machine learning models "
        "on the synced data.",
    )
    _add_connection_arguments(sync_parser, suppress_defaults=True)
    _add_database_argument(sync_parser, suppress_defaults=True)
    _add_identification_argument(sync_parser, suppress_defaults=True)
    sync_parser.add_argument(
        "--quick",
        action="store_true",
        help="Only add new emails, skip re-checking labels of already-known emails.",
    )
    sync_parser.add_argument(
        "--folder",
        action="append",
        metavar="FOLDER",
        help="Restrict the sync to this IMAP folder - may be given multiple times; "
        "default: every folder.",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train/retrain the machine learning models from the local database.",
        description="Train one machine learning model per folder on the emails already "
        "stored in the local database (see 'mailsort sync'). Does not connect to the mail "
        "server and does not move any messages.",
    )
    _add_database_argument(train_parser, suppress_defaults=True)
    _add_identification_argument(train_parser, suppress_defaults=True)
    train_parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include emails marked as deleted when training.",
    )
    train_parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip probability calibration and keep raw classifier scores even where there is "
        "enough data to calibrate. Calibration is applied automatically per folder where there "
        "is enough data (see 'mailsort evaluate'); this only turns it off entirely.",
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Report machine learning recommendations for a folder without moving anything.",
        description="Download the messages currently in FOLDER and report the machine "
        "learning model's recommendation for each - a read-only preview of 'mailsort sort'. "
        "Nothing is moved, deleted or otherwise modified on the mail server.",
    )
    predict_parser.add_argument(
        "folder", metavar="FOLDER", help="IMAP folder to fetch and score messages from."
    )
    _add_connection_arguments(predict_parser, suppress_defaults=True)
    _add_database_argument(predict_parser, suppress_defaults=True)
    _add_identification_argument(predict_parser, suppress_defaults=True)
    _add_recommendation_arguments(predict_parser)

    sort_parser = subparsers.add_parser(
        "sort",
        help="Classify messages in a folder and move the ones above the threshold.",
        description="Download the messages currently in FOLDER, score them with the "
        "trained machine learning model, and move the ones whose score clears "
        "--recommendation-ratio to the recommended folder. See 'mailsort predict' to preview "
        "this without moving anything.",
    )
    sort_parser.add_argument(
        "folder",
        metavar="FOLDER",
        help="IMAP folder to fetch, score and sort messages from.",
    )
    _add_connection_arguments(sort_parser, suppress_defaults=True)
    _add_database_argument(sort_parser, suppress_defaults=True)
    _add_identification_argument(sort_parser, suppress_defaults=True)
    _add_recommendation_arguments(sort_parser)

    status_parser = subparsers.add_parser(
        "status",
        help="Report information about the local database.",
        description="Report information about the local database - its location, how many "
        "messages it knows about, and how many per-folder machine learning models have been "
        "trained. Does not connect to the mail server.",
    )
    _add_database_argument(status_parser, suppress_defaults=True)
    _add_identification_argument(status_parser, suppress_defaults=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Estimate precision/recall/coverage for a recommendation-ratio, on held-out data.",
        description="Train a separate, throwaway set of per-folder models on part of the local "
        "database and score them against the rest, to estimate precision, recall, F1, support "
        "and coverage/abstention rate at --recommendation-ratio - the evidence a "
        "recommendation-ratio choice should be based on, rather than treating "
        "recommendation-ratio as if it were already a calibrated probability. Does not connect "
        "to the mail server, and never changes the models 'mailsort train' has already stored.",
    )
    _add_database_argument(evaluate_parser, suppress_defaults=True)
    _add_identification_argument(evaluate_parser, suppress_defaults=True)
    evaluate_parser.add_argument(
        "--recommendation-ratio",
        type=float,
        default=_DEFAULT_RECOMMENDATION_RATIO,
        help="Cutoff a score must clear to count as accepted (0<r<1) - default: "
        f"{_DEFAULT_RECOMMENDATION_RATIO} . Ignored when --sweep is given.",
    )
    evaluate_parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of email threads held out for testing rather than training "
        "(0<test_size<1) - default: 0.25 .",
    )
    evaluate_parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Evaluate with calibration turned off, to compare against the default.",
    )
    evaluate_parser.add_argument(
        "--sweep",
        action="store_true",
        help="Report coverage and precision at several recommendation-ratio values "
        "(0.5, 0.7, 0.8, 0.9, 0.95, 0.99) instead of a single detailed report, to help pick a "
        "threshold.",
    )

    return parser


def _parse_db_user_id(identification):
    if not identification:
        return 1
    try:
        return int(identification)
    except ValueError:
        raise ValueError(
            f"-i/--identification must be an integer, got {identification!r}."
        ) from None


def _missing_connection_fields(args):
    missing = []
    if not args.host:
        missing.append("--host")
    if not args.username:
        missing.append("--username")
    if not args.password:
        missing.append("--password")
    return missing


def _connect(args, database, db_user_id):
    return Imap(
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password,
        connection_str=database,
        db_user_id=db_user_id,
        use_ssl=not args.no_ssl,
        email_download_format="metadata",
    )


def _report_missing_connection_fields(command, missing):
    print(
        f"mailsort {command}: missing required argument(s): {', '.join(missing)}",
        file=sys.stderr,
    )
    return _EXIT_CONFIG_ERROR


def _run_sync(args, database, db_user_id):
    missing = _missing_connection_fields(args)
    if missing:
        return _report_missing_connection_fields("sync", missing)
    imap = _connect(args, database, db_user_id)
    try:
        imap.update_database(quick=args.quick, label_lst=args.folder)
    finally:
        imap.close()
    print(f"Synced local database at {database!r}.")
    return _EXIT_OK


def _run_train(args, database, db_user_id):
    model_count = train_machine_learning_models(
        connection_str=database,
        db_user_id=db_user_id,
        include_deleted=args.include_deleted,
        calibrate=not args.no_calibration,
    )
    print(f"Trained {model_count} folder model(s) and stored them in {database!r}.")
    return _EXIT_OK


def _run_predict(args, database, db_user_id):
    missing = _missing_connection_fields(args)
    if missing:
        return _report_missing_connection_fields("predict", missing)
    imap = _connect(args, database, db_user_id)
    try:
        recommendations = imap.get_label_recommendations(
            label=args.folder,
            recommendation_ratio=args.recommendation_ratio,
            label_prefix=args.label_prefix,
        )
    finally:
        imap.close()
    print(_format_recommendations_table(recommendations))
    return _EXIT_OK


def _run_sort(args, database, db_user_id):
    missing = _missing_connection_fields(args)
    if missing:
        return _report_missing_connection_fields("sort", missing)
    imap = _connect(args, database, db_user_id)
    try:
        result = imap.filter_messages_from_server(
            label=args.folder,
            recommendation_ratio=args.recommendation_ratio,
            label_prefix=args.label_prefix,
        )
    finally:
        imap.close()
    print(f"Sorted folder {args.folder!r}: moved {result.moved_count} message(s).")
    return _EXIT_OK


def _run_status(args, database, db_user_id):
    status = get_database_status(connection_str=database, db_user_id=db_user_id)
    print(_format_status_table(status))
    return _EXIT_OK


_SWEEP_RECOMMENDATION_RATIO_LST = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


def _run_evaluate(args, database, db_user_id):
    holdout = score_holdout(
        connection_str=database,
        db_user_id=db_user_id,
        test_size=args.test_size,
        calibrate=not args.no_calibration,
    )
    if holdout.test_message_count == 0:
        print(
            "Not enough data to evaluate: the held-out test split has no scoreable messages. "
            "Sync more mail, or lower --test-size."
        )
        return _EXIT_OK
    if args.sweep:
        reports = evaluate_thresholds(holdout, list(_SWEEP_RECOMMENDATION_RATIO_LST))
        print(_format_evaluation_sweep(reports))
    else:
        report = evaluate_at_threshold(
            holdout, recommendation_ratio=args.recommendation_ratio
        )
        print(_format_evaluation_report(report))
    return _EXIT_OK


_COMMAND_HANDLERS = {
    "sync": _run_sync,
    "train": _run_train,
    "predict": _run_predict,
    "sort": _run_sort,
    "status": _run_status,
    "evaluate": _run_evaluate,
}


def command_line_parser(argv=None):
    """
    Entry point for the mailsort command line interface.

    Args:
        argv (list/None): argument vector to parse, defaults to sys.argv[1:]

    Returns:
        int: process exit code - 0 on success, 2 for invalid configuration, 1 for a runtime
            error raised while executing the command
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        db_user_id = _parse_db_user_id(getattr(args, "identification", None))
    except ValueError as error:
        print(f"mailsort: error: {error}", file=sys.stderr)
        return _EXIT_CONFIG_ERROR

    database = getattr(args, "database", None) or _DEFAULT_DATABASE
    handler = _COMMAND_HANDLERS.get(args.command)

    if handler is None:
        parser.print_help()
        return _EXIT_OK

    try:
        return handler(args=args, database=database, db_user_id=db_user_id)
    except Exception as error:
        print(f"mailsort: error: {error}", file=sys.stderr)
        return _EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(command_line_parser())
