import argparse

from mailsort import Imap

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


def command_line_parser():
    """
    Main function primarily used for the command line interface of the IMAP backend
    """
    parser = argparse.ArgumentParser(prog="mailsort")
    parser.add_argument(
        "--host",
        help="IMAP server hostname e.g. imap.example.com .",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=993,
        help="IMAP server port - default: 993 .",
    )
    parser.add_argument(
        "--username",
        help="IMAP account username.",
    )
    parser.add_argument(
        "--password",
        help="IMAP account password.",
    )
    parser.add_argument(
        "--no-ssl",
        action="store_true",
        help="Connect without SSL (IMAP4 instead of IMAP4_SSL).",
    )
    parser.add_argument(
        "-d",
        "--database",
        help="Connection string to connect to database e.g. sqlite:///email.db .",
    )
    parser.add_argument(
        "-u",
        "--update",
        action="store_true",
        help="Update local database and retrain machine learning model.",
    )
    parser.add_argument(
        "-i",
        "--identification",
        help="User ID of the database user e.g. 1 .",
    )
    parser.add_argument(
        "-l",
        "--label",
        help="Email label (IMAP folder) to be filtered with machine learning.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help=(
            "With -l/--label, print the machine learning recommendations for that folder "
            "instead of moving any messages."
        ),
    )
    args = parser.parse_args()
    db_user_id = int(args.identification) if args.identification else 1
    if not args.host or not args.username:
        print("Please provide --host and --username.")
    elif not args.password:
        print("Please provide --password.")
    else:
        database = args.database or "sqlite:///email.db"
        imap = Imap(
            host=args.host,
            port=args.port,
            username=args.username,
            password=args.password,
            connection_str=database,
            db_user_id=db_user_id,
            use_ssl=not args.no_ssl,
            email_download_format="metadata",
        )
        if args.update:
            imap.update_database(quick=False)
            imap.fit_machine_learning_model_to_database(
                n_estimators=100,
                max_features=400,
                random_state=42,
                bootstrap=True,
                include_deleted=False,
            )
        elif args.label and args.dry_run:
            recommendations = imap.get_label_recommendations(
                label=args.label, recommendation_ratio=0.9, label_prefix="labels_"
            )
            print(_format_recommendations_table(recommendations))
        elif args.label:
            imap.filter_messages_from_server(
                label=args.label, recommendation_ratio=0.9, label_prefix="labels_"
            )
        else:
            parser.print_help()


if __name__ == "__main__":
    command_line_parser()
