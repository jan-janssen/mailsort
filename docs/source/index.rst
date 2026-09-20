.. mailsort documentation master file, created by
   sphinx-quickstart.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Similarity based email sorting for any IMAP mail server
=========================================================
`mailsort` reduces the interruption caused by the increase of daily emails by automatically sorting your emails into 
folders (IMAP labels) based on the similarity of new messages to the ones you have already sorted. It uses machine 
learning to learn from your existing folder assignments and suggests or applies folders. 

This folder based approach allows you to focus on the most important emails first, while still being able to access less
important emails later. You are in contorl. When you change the folder assignments, you can retrain the `mailsort` 
machine learning model so `mailsort` can adapt to your changing preferences. Think of it as a personal assistant that 
helps you manage your emails.

`mailsort` is deliberately a plain Python library and command line tool rather than a hosted service: you run it
yourself, on your own schedule (for example from cron), against your own local database. Your emails belong to you.

`mailsort` connects to any IMAP mail server, stores your emails locally in an SQLite database, and trains a machine 
learning model on the folders (IMAP labels) you have already assigned to your emails, and uses that model to suggest or 
apply folders to new messages. 

Motivation
----------
In 2020 there were `306.4 billion e-mails <https://www.statista.com/statistics/456500/daily-number-of-e-mails-worldwide/>`_
sent and received daily. This number is estimated to increase by 4% yearly, resulting in over 376.4 billion e-mails by
2025. While email as medium for internal communication in large enterprises is slowly replaced by instant messaging
solutions and business communication platforms, these solutions fail to address the primary challenge, namely the
communication between employees from different companies. So addressing the `stress and productivity lost <https://affect.media.mit.edu/pdfs/16.Mark-CHI_Email.pdf>`_
resulting from interruptions caused by the increase of daily emails is the motivation for the development of mailsort.

Documentation
-------------

.. toctree::
   :maxdepth: 2

   preparation
   configuration
   architecture
   evaluation
   troubleshooting
   support
   developer
