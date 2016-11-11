import getopt, csv
import got3 as got

querysearch = 'Brexit'
# maxtweets=
since = '2016-06-20'
until = '2016-06-25'
try:
    # opts, args = getopt.getopt(argv, "", ("username=", "since=", "until=", "querysearch=", "toptweets", "maxtweets="))

    tweetCriteria = got.manager.TweetCriteria()

    tweetCriteria.since = since

    tweetCriteria.until = until

    tweetCriteria.querySearch = querysearch

    tweetCriteria.topTweets = True

    tweetCriteria.lang = 'en'
    # tweetCriteria.maxTweets = int(arg)

    outputFile = open("output_got.csv", "+w")
    writer = csv.writer(outputFile)
    writer.writerow(['wordnouns','geo','polarity','subjectivity','time','hashtags'])
    # writer.writerow(['id', 'time', 'text', 'geo', 'hashtags'])
    print('Searching...')


    def receiveBuffer(tweets):
        for t in tweets:
            # writer.writerow([t.date.strftime("%Y-%m-%d %H:%M"), t.text, t.geo, t.hashtags])
        # (t.date.strftime("%Y-%m-%d %H:%M"), t.text, t.geo,t.hashtags)))
            writer.writerow([t.wordnouns, t.geo, t.polarity, t.subjectivity, t.date.strftime("%Y-%m-%d %H:%M"), t.hashtags])
        outputFile.flush()
        print('{} Tweets saved on file...'.format(len(tweets)))


    got.manager.TweetManager.getTweets(tweetCriteria, receiveBuffer)

except Exception as inst:
    print('Arguments parser error, try -h' + inst.args)
finally:
    outputFile.close()
    print('Done. Output file generated "output_got.csv".')
