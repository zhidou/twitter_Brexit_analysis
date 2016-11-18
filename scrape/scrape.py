import csv, time
import got3 as got
from helper import load_data
import os

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
        if outputFile.closed:
            outputFile = open("Tweets" + tweetCriteria.month + ".csv", "+a")
        try:
            got.manager.getTweets(tweetCriteria, receiveBuffer)
        except KeyboardInterrupt:
            print('KeyboardInterrupt stop ' + criteria['month'])
            if not criteria.get('pid'):
                raise KeyboardInterrupt
            break

        except Exception as inst:
            if len(inst.args) > 0:
                print(inst.args[0])
            print(tweetCriteria.month + ' Error happens! Arguments parser error:' + str(inst.args))
            if Error_time < 3:
                print("sleep 300s and retry!")
                time.sleep(3)
                criteria = load_data([criteria['month']])[0]
                tweetCriteria.refreshCursor = criteria['refreshCursor']
                Error_time += 1
                pass
            else:
                print("reconnect 3 times, but still fail " + tweetCriteria.month)
                if not criteria.get('pid'):
                    raise Exception(inst)
                break
        else:
            print('Data of ' + tweetCriteria.month + ' has completely downloaded!!')
            break
        finally:
            outputFile.close()
            print(tweetCriteria.month + 'Running time: {}'.format(time.time() - beginTime))
    return

