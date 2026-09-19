# Preparation
`mailsort` is a Python module to automate the filtering of emails on any IMAP mail server. It assigns emails to
folders based on their similarity to other emails already assigned to the same folder.

Many people struggle with the increasing email volume leading to hundreds of unread emails. As the capabilities of
even the best search engine are limited when it comes to large numbers of emails, the only way to keep an overview
is filtering emails into folders. The manual process of filtering emails into folders is tedious, still most people
are too lazy to create email filters and keep their email filters up to date. Finally, in the age of mobile
computing when most people access their emails from their smartphone, the challenge of sorting emails is more
relevant than ever.

The solution to this challenge is to automatically filter emails depending on their similarity to existing emails in
a given folder. This solution was already proposed in a couple of research papers ranging from the filtering of spam
emails [E.G. Dada et al.](https://doi.org/10.1016/j.heliyon.2019.e01802) to the specific case of sorting emails into
folders [R. Bekkerman et al.](https://people.cs.umass.edu/~mccallum/papers/foldering-tr05.pdf). Even a couple of
opensource prototypes are available like [ml-email-clustering](https://github.com/anthdm/ml-email-clustering) and
[emailinsight](https://github.com/andreykurenkov/emailinsight).

`mailsort` takes this idea and applies it to any mail server reachable over plain IMAP - so it works equally well
with a self-hosted mail server, a company mailbox, or any provider that is not Google Mail (for Google Mail itself,
see [gmailsorter](https://github.com/jan-janssen/gmailsorter)). Before using `mailsort` you first have to sort your
emails manually. This step is necessary for `mailsort` to learn your preferences.

## Sort your emails
While the ordering of information is generally a very personal topic, with no one solution which fits anybody. Still
some general considerations can be helpful to sort your emails:

* How many emails go into one email folder? And how many email folders do you need? It is typically suggested to
  start with around ten email folders with each folder containing more or less the same amount of emails.
* To sort your email folders you can start their names with a number, this enforces the right order of email folders
  independent of the email client.
* For `mailsort` to be able to learn how to treat any kind of incoming email, it makes sense to have an email folder
  for emails you plan to delete. These could be newsletters, spam messages or other unrelated emails.

Before you activate `mailsort` it is essential that you clean up your whole inbox and sort each email into a folder.
Do not delete the emails even if they are irrelevant. To sort your emails you can use your favorite email client, or
alternatively you can log in to your mail provider's web interface and sort your emails there.

## Configure your email account
For `mailsort` to connect to your mail account, it needs an IMAP server hostname, port, username and password.

### Enable IMAP access
Most mail providers disable IMAP access by default and require you to explicitly enable it in your account
settings - check your provider's documentation for the exact steps (searching for "<your provider> enable IMAP" is
usually the fastest way to find them).

### Use an app password instead of your account password
If your mail provider supports two-factor authentication, it typically requires you to create a separate app
password for IMAP access rather than using your regular account password. This has the added benefit that the app
password can be revoked independently at any time, without having to change your main account password. `mailsort`
never persists your password anywhere - it is only used to log in to the IMAP server for the duration of a
connection.

### Create one separate folder as inbox for mailsort
To minimize the interruption of incoming emails, it is recommended to create a new inbox folder, for example named
`MailSortInbox`. You then configure a filter in your mail account (or your existing email client) to redirect new
incoming emails into this folder. Emails in the `MailSortInbox` folder are then periodically sorted using
`mailsort`, typically every five minutes - see [Configuration](configuration) for how to schedule this.
