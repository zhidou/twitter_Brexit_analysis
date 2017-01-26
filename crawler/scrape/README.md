This scrape is modified based on https://github.com/Jefferson-Henrique/GetOldTweets-python

The modified we have:
1. For our project, we want to search tweets with hash tag: Brexit (and others)
2. Now the code supports multiple processes
3. Now supports resume and stop at breakpoints without losing integrity
4. If server stop provide service (visit too many times, and like that), it will sleep and restart automatically

You could set the month and hashtag to you want in main function.
