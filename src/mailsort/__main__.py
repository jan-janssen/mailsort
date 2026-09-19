"""
Command line interface for mailsort.

This module only parses arguments and formats output - the actual work is done by
mailsort.Imap and the reusable functions in mailsort.training / mailsort.status.
"""

import argparse
import sys

from mailsort import Imap
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
    "recommended_label": 20,
}


def _format_recommendations_table(recommendations):
    """
    Render the list of dicts returned by Imap.get_label_recommendations() as a plain text
    table for the command line - one row per message, no messages are moved to build this.

    Args:
        recommendations (list): return value of AbstractMailBox.get_label_recommendations()

    Returns:
        str: human-readable table, or a placeholder message if there are no messages
    """
    if not recommendations:
        return "No messages found in this folder."
    widths = _TABLE_COLUMN_WIDTHS
    header = (
        f"{'MESSAGE ID':<{widths['message_id']}} "
        f"{'SUBJECT':<{widths['subject']}} "
        f"{'RECOMMENDED LABEL':<{widths['recommended_label']}} "
        f"{'SCORE':>6} {'REACHED':>8}"
    )
    lines = [header, "-" * len(header)]
    for entry in recommendations:
        message_id = str(entry["message_id"])[: widths["message_id"]]
        subject = str(entry["subject"] or "")[: widths["subject"]]
        recommended_label = str(entry["recommended_label"] or "-")[
            : widths["recommended_label"]
        ]
        lines.append(
            f"{message_id:<{widths['message_id']}} "
            f"{subject:<{widths['subject']}} "
            f"{recommended_label:<{widths['recommended_label']}} "
            f"{entry['score']:>6.2f} {str(entry['threshold_reached']):>8}"
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

Run `mailsort <command> --help` for the arguments of an individual command.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Top-level flags kept for backwards compatibility with the pre-subcommand CLI
    # (deprecated - prefer the sync/train/predict/sort/status subcommands below).
    _add_connection_arguments(parser)
    _add_database_argument(parser)
    _add_identification_argument(parser)
    parser.add_argument(
        "-u",
        "--update",
        action="store_true",
        help="Deprecated - use 'mailsort sync' followed by 'mailsort train' instead. "
        "Update local database and retrain machine learning model.",
    )
    parser.add_argument(
        "-l",
        "--label",
        help="Deprecated - use 'mailsort predict FOLDER' or 'mailsort sort FOLDER' instead. "
        "Email label (IMAP folder) to be filtered with machine learning.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Deprecated - use 'mailsort predict FOLDER' instead. With -l/--label, print "
        "the machine learning recommendations for that folder instead of moving any messages.",
    )

    subparsers = parser.add_subparsers(
        dest="command", metavar="{sync,train,predict,sort,status}"
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
        "folder", metavar="FOLDER", help="IMAP folder to fetch, score and sort messages from."
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
        imap.filter_messages_from_server(
            label=args.folder,
            recommendation_ratio=args.recommendation_ratio,
            label_prefix=args.label_prefix,
        )
    finally:
        imap.close()
    print(f"Sorted folder {args.folder!r}.")
    return _EXIT_OK


def _run_status(args, database, db_user_id):
    status = get_database_status(connection_str=database, db_user_id=db_user_id)
    print(_format_status_table(status))
    return _EXIT_OK


def _run_legacy(args, parser, database, db_user_id):
    """
    Pre-subcommand CLI, kept for backwards compatibility with existing scripts using
    -u/--update or -l/--label - see sync/train/predict/sort for the replacements.
    """
    if not args.host or not args.username:
        print("Please provide --host and --username.")
        return _EXIT_CONFIG_ERROR
    if not args.password:
        print("Please provide --password.")
        return _EXIT_CONFIG_ERROR
    if args.update:
        print(
            "mailsort: -u/--update is deprecated, use 'mailsort sync' followed by "
            "'mailsort train' instead.",
            file=sys.stderr,
        )
        imap = _connect(args, database, db_user_id)
        try:
            imap.update_database(quick=False)
            imap.fit_machine_learning_model_to_database(
                n_estimators=100,
                max_features=400,
                random_state=42,
                bootstrap=True,
                include_deleted=False,
            )
        finally:
            imap.close()
        return _EXIT_OK
    elif args.label and args.dry_run:
        print(
            "mailsort: -l/--label with -n/--dry-run is deprecated, use "
            "'mailsort predict FOLDER' instead.",
            file=sys.stderr,
        )
        imap = _connect(args, database, db_user_id)
        try:
            recommendations = imap.get_label_recommendations(
                label=args.label,
                recommendation_ratio=_DEFAULT_RECOMMENDATION_RATIO,
                label_prefix=_DEFAULT_LABEL_PREFIX,
            )
        finally:
            imap.close()
        print(_format_recommendations_table(recommendations))
        return _EXIT_OK
    elif args.label:
        print(
            "mailsort: -l/--label is deprecated, use 'mailsort sort FOLDER' instead.",
            file=sys.stderr,
        )
        imap = _connect(args, database, db_user_id)
        try:
            imap.filter_messages_from_server(
                label=args.label,
                recommendation_ratio=_DEFAULT_RECOMMENDATION_RATIO,
                label_prefix=_DEFAULT_LABEL_PREFIX,
            )
        finally:
            imap.close()
        return _EXIT_OK
    else:
        parser.print_help()
        return _EXIT_OK


_COMMAND_HANDLERS = {
    "sync": _run_sync,
    "train": _run_train,
    "predict": _run_predict,
    "sort": _run_sort,
    "status": _run_status,
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

    try:
        if handler is not None:
            return handler(args=args, database=database, db_user_id=db_user_id)
        return _run_legacy(
            args=args, parser=parser, database=database, db_user_id=db_user_id
        )
    except Exception as error:
        print(f"mailsort: error: {error}", file=sys.stderr)
        return _EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(command_line_parser())
