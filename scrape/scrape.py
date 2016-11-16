import csv, time
from .got3 import manager
from .helper import load_data

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
    tweetCriteria = manager.TweetCriteria()
    setcriteria(criteria, tweetCriteria)
    print("Begin to scrape data from " + tweetCriteria.month)

    outputFile = open("Tweets" + tweetCriteria.month + ".csv", "+a")
    writer = csv.writer(outputFile)
    writer.writerow(['user_id', 'time', 'geo', 'polarity', 'subjectivity', 'wordnouns', 'hashtags'])
    print('Downloading data of ' + tweetCriteria.month + '...')

    def receiveBuffer(tweets):
        for t in tweets:
            writer.writerow([t.user_id, t.date.strftime("%Y-%m-%d %H:%M"), t.geo,
                             t.polarity, t.subjectivity, t.wordnouns, t.hashtags])
        outputFile.flush()

    Error_time = 0
    while True:
        try:
            manager.getTweets(tweetCriteria, receiveBuffer)
        except KeyboardInterrupt:
            print('KeyboardInterrupt stop ' + criteria['month'])
            raise KeyboardInterrupt
        except Exception as inst:
            if len(inst.args) > 0:
                print(inst.args[0])
            print(tweetCriteria.month + ' Error happens! Arguments parser error:' + str(inst.args))
            if Error_time < 3:
                print("sleep 300s and retry!")
                time.sleep(300)
                month = [x['month'] for x in criteria]
                criteria = load_data(month)
                Error_time += 1
                pass
            else:
                print("fail!!!!")
                raise Exception(inst)
        else:
            print('Data of ' + tweetCriteria.month + ' has completely downloaded!!')
            break
        finally:
            outputFile.close()
            print('Running time: {}'.format(time.time() - beginTime))
    return

