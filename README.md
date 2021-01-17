# IO2-rwq

Well I read too late that we were supposed to attach a repository link to the report instead of attaching code files... So I wouldn't call it a repository now, more like an upload folder, sadly.
Hopefully it's not a problem.

This project was to compare different classifiers and how good they were at guessing red wines' quality. There are two versions (as ~62% accuracy of the first version wasn't satisfying for me):

1. "quality-0-10.py", where we're trying to guess 0-10 wine rating from 'quality' column.
2. "is-good-0-1.py", where we're just trying to guess if the wine is good or not. A new column 'is-good' is created where, if value in 'quality' is 7 or higher, a value 1 is added. Otherwise 0 is added. Accuracy improves mostly because there's only a 0-1 guess, not 0-10.

Dataset source: https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009