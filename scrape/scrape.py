import csv, time
import got3 as got
from helper import load_data
<<<<<<< HEAD
import os
=======
>>>>>>> e25ab4dfd22d84c16f64a5d50f3ddcb6152cbca1

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
<<<<<<< HEAD
=======

>>>>>>> e25ab4dfd22d84c16f64a5d50f3ddcb6152cbca1
    outputFile = open("Tweets" + tweetCriteria.month + ".csv", "+a")
    writer = csv.writer(outputFile)
    writer.writerow(['user_id', 'time', 'geo', 'polarity', 'subjectivity', 'wordnouns', 'hashtags'])
    print('Downloading data of ' + tweetCriteria.month + '...')
<<<<<<< HEAD
=======

>>>>>>> e25ab4dfd22d84c16f64a5d50f3ddcb6152cbca1
    def receiveBuffer(tweets):
        for t in tweets:
            writer.writerow([t.user_id, t.date.strftime("%Y-%m-%d %H:%M"), t.geo,
                             t.polarity, t.subjectivity, t.wordnouns, t.hashtags])
        outputFile.flush()

    Error_time = 0
    while True:
        try:
            got.manager.getTweets(tweetCriteria, receiveBuffer)
        except KeyboardInterrupt:
            print('KeyboardInterrupt stop ' + criteria['month'])
<<<<<<< HEAD
            if not criteria.get('pid'):
                raise KeyboardInterrupt
            break

=======
            raise KeyboardInterrupt
>>>>>>> e25ab4dfd22d84c16f64a5d50f3ddcb6152cbca1
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
<<<<<<< HEAD
                print("reconnect 3 times, but still fail " + tweetCriteria.month)
                if not criteria.get('pid'):
                    raise Exception(inst)
                break
=======
                print("fail!!!!")
                raise Exception(inst)
>>>>>>> e25ab4dfd22d84c16f64a5d50f3ddcb6152cbca1
        else:
            print('Data of ' + tweetCriteria.month + ' has completely downloaded!!')
            break
        finally:
            outputFile.close()
            print('Running time: {}'.format(time.time() - beginTime))
    return

