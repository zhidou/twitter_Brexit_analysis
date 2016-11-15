import csv, time
import got3 as got


def setcriteria(criteria, tweetCriteria):
    tweetCriteria.since = criteria['since']
    tweetCriteria.until = criteria['until']
    tweetCriteria.querySearch = criteria['querysearch']
    tweetCriteria.topTweets = criteria['topTweets']
    tweetCriteria.lang = criteria['lang']
    tweetCriteria.refreshCursor = criteria['refreshCursor']
    tweetCriteria.month = criteria['month']
    tweetCriteria.num = criteria['num']
    tweetCriteria.dic = criteria



def scrape(criteria):
    beginTime = time.time()
    tweetCriteria = got.manager.TweetCriteria()
    setcriteria(criteria, tweetCriteria)
    print("Begin to scrape data from " + tweetCriteria.month)
    try:
        outputFile = open("Tweets" + tweetCriteria.month + ".csv", "+a")
        writer = csv.writer(outputFile)
        writer.writerow(['user_id', 'time', 'geo', 'polarity', 'subjectivity', 'wordnouns', 'hashtags'])
        print('Downloading data of ' + tweetCriteria.month + '...')

        def receiveBuffer(tweets):
            for t in tweets:
                writer.writerow([t.user_id, t.date.strftime("%Y-%m-%d %H:%M"), t.geo,
                                 t.polarity, t.subjectivity, t.wordnouns,t.hashtags])
            outputFile.flush()

        got.manager.getTweets(tweetCriteria, receiveBuffer)
    except Exception as inst:
        print(inst.args[0])
        print(tweetCriteria.month + ' Error happens! Arguments parser error:' + str(inst.args))
        raise Exception(inst)
    else:
        print('Data of' + tweetCriteria.month + ' has completely downloaded!!')
    finally:
        outputFile.close()
        print('Running time: {}'.format(time.time() - beginTime))
