import csv, time
import got3 as got
from multiprocessing import Pool

def multiprocess(data):
    t = time.time()
    if len(data) == 1:
        download(data[0])
    else:
        pool = Pool(processes=len(data))
        pool.map(download, data)
        pool.close()
        pool.join()
    print("Download {0} month tweets! Running time {1}".format(len(data), time.time() - t))


def download(criteria):
    month = criteria[1]
    tweetCriteria = criteria[0]
    manager = got.manager.TweetManager()
    try:
        outputFile = open("Tweets" + month + ".csv", "+a")
        writer = csv.writer(outputFile)
        writer.writerow(['user_id', 'time', 'geo', 'polarity', 'subjectivity', 'wordnouns', 'hashtags'])
        # writer.writerow(['id', 'time', 'text', 'geo', 'hashtags'])
        print('Downloading data of ' + month + '...')

        def receiveBuffer(tweets):
            for t in tweets:
                # writer.writerow([t.date.strftime("%Y-%m-%d %H:%M"), t.text, t.geo, t.hashtags])
                # (t.date.strftime("%Y-%m-%d %H:%M"), t.text, t.geo,t.hashtags)))
                writer.writerow([t.user_id, t.date.strftime("%Y-%m-%d %H:%M"), t.geo,
                                 t.polarity, t.subjectivity, t.wordnouns,t.hashtags])
            outputFile.flush()

        # got.manager.TweetManager.getTweets(criteria, receiveBuffer)
        manager.getTweets(criteria, receiveBuffer)

    except Exception as inst:
        print('Arguments parser error:')
        print(inst.args)
    else:
        print('Data of' + month + ' has downloaded!!')
    finally:
        outputFile.close()

querysearch = 'Brexit'
since6 = '2016-06-01'
until6 = '2016-07-01'
tweetCriteria6 = got.manager.TweetCriteria()
tweetCriteria6.since = since6
tweetCriteria6.until = until6
tweetCriteria6.querySearch = querysearch
tweetCriteria6.topTweets = True
tweetCriteria6.lang = 'en'
tweetCriteria6.refreshCursor =''

since7 = '2016-07-01'
until7 = '2016-08-01'
tweetCriteria7 = got.manager.TweetCriteria()
tweetCriteria7.since = since7
tweetCriteria7.until = until7
tweetCriteria7.querySearch = querysearch
tweetCriteria7.topTweets = True
tweetCriteria7.lang = 'en'

since8 = '2016-08-01'
until8 = '2016-09-01'
tweetCriteria8 = got.manager.TweetCriteria()
tweetCriteria8.since = since8
tweetCriteria8.until = until8
tweetCriteria8.querySearch = querysearch
tweetCriteria8.topTweets = True
tweetCriteria8.lang = 'en'


multiprocess([[tweetCriteria6, 'June']])
# tweetCriteria.maxTweets = int(arg)
# multiprocess([[tweetCriteria6, 'June'],[tweetCriteria7, 'July'], [tweetCriteria8, 'Aug']])
