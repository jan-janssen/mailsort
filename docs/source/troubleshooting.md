# Troubleshooting
`mailsort` is under active development, so things occasionally go wrong. This page collects fixes for the problems
users run into most often. If your problem is not listed here, please
[open a new issue](https://github.com/jan-janssen/mailsort/issues/new) - real questions become the next entry on
this page.

## `imaplib.IMAP4.error: b'LOGIN failed.'` or similar login errors
This almost always means one of the following:

* **Two-factor authentication is enabled** on your mail account, and your provider expects a dedicated app password
  rather than your regular account password for IMAP access - see [Preparation](preparation) for how to create one.
* **IMAP access is disabled** in your account settings. Most providers disable it by default and require you to
  explicitly turn it on.
* The **username** is not what you expect - some providers require the full email address as the username, others
  a separate account name.

## Connection errors, timeouts, or SSL certificate errors
* Double-check `--host` and `--port` (or `host`/`port` in the Python interface) against your provider's documented
  IMAP settings - the most common port is `993` for `IMAP4_SSL` (the default) and `143` for plain IMAP4.
* If your provider only supports plain IMAP4 without SSL, pass `--no-ssl` on the command line, or `use_ssl=False`
  in the Python interface. Do this only on a trusted network, since the connection - including your password - is
  then sent unencrypted.

## A folder is never suggested, even after training
`mailsort` deliberately never trains on, or recommends moving mail into, folders marked `\Noselect`, or special-use
folders such as `Trash`, `Spam`/`Junk`, `Sent` and `Drafts` (see [How mailsort works](architecture)). If a folder
you expect to see suggestions for is one of these, or a folder your mail server marks similarly, this is expected
behaviour, not a bug.

Beyond that, a brand-new folder with very few emails filed into it will rarely reach the default 90%
`recommendation_ratio` needed to be suggested - keep sorting a few more emails into it by hand and retrain
(`-u`/`update_database()` followed by `fit_machine_learning_model_to_database()`) before expecting suggestions.

## `Could not move IMAP message ... to ...`
This means the IMAP server rejected the `MOVE` (or fallback `COPY` + `STORE \Deleted`) command `mailsort` issued to
file an email into the recommended folder. This is usually caused by:

* The target folder no longer existing, or having been renamed, on the server.
* Insufficient permissions on the target folder (for example a shared or read-only folder).

## Still stuck?
Please [open an issue on GitHub](https://github.com/jan-janssen/mailsort/issues) with as much detail as you can
provide - your mail provider, the exact command or Python call you used (with the password redacted), and the
exact error message you see. See [Support](support) for how the project is maintained.
