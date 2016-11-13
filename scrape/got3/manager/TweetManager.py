import urllib.request, urllib.parse, urllib.error, urllib.request, urllib.error,urllib.parse,json,re,datetime,sys,http.cookiejar
from textblob import TextBlob
from got3 import models
from pyquery import PyQuery
import time
import numpy as np


class TweetManager:
    def __init__(self):
        pass

    # @staticmethod
    def getTweets(self, criteria, receiveBuffer = None, bufferLength = 100):
        tweetCriteria = criteria[0]
        month = criteria[1]
        refreshCursor = tweetCriteria.refreshCursor

        counter = 0
        output_counter = 0
        resultsAux = []
        cookieJar = http.cookiejar.CookieJar()

        active = True
        urlInfo = self.generateUrl(tweetCriteria)

        while active:

            tweets = []
            for i in range(4):
                json = self.getJsonReponse(urlInfo, refreshCursor, cookieJar, month)
                if len(json['items_html'].strip()) == 0:
                    break
                refreshCursor = json['min_position']
                tweets_html = PyQuery(json['items_html'])('div.js-stream-tweet')

                tweets.append(tweets_html[np.random.randint(0, len(tweets_html))])

            if len(tweets) == 0:
                break

            tweetHTML = tweets[np.random.randint(0, len(tweets))]

            tweetPQ = PyQuery(tweetHTML)
            tweet = models.Tweet()

            # process text
            txt = re.sub(r"\s+", " ", tweetPQ("p.js-tweet-text").text().replace('# ', '#').replace('@ ', '@'))
            txt = re.sub(r"http[s]?://.(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+", '', txt)
            txt = re.sub(r'(?:@[\w_]+)', '', txt)
            txt = re.sub(r'[\w]+/(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]\s))+', '', txt)
            txt = re.sub(r'(?:./[0-9]+)', '', txt)
            txt = txt.replace(',', '')
            txt = txt.replace('/ ', '')
            txt1 = txt
            txt = re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", '', txt)

            blob = TextBlob(txt)
            polarity, subjectivity = blob.sentiment

            dateSec = int(tweetPQ("small.time span.js-short-timestamp").attr("data-time"))

            id = tweetPQ.attr("data-tweet-id")
            usr_id = int(tweetPQ("a.js-user-profile-link").attr("data-user-id"))

            geo = ''
            geoSpan = tweetPQ('span.Tweet-geo')
            if len(geoSpan) > 0:
                geo = geoSpan.attr('title')

            tweet.id = id
            tweet.user_id = usr_id
            tweet.wordnouns = (",").join(blob.noun_phrases)
            tweet.polarity = polarity
            tweet.subjectivity= subjectivity
            tweet.text = txt
            tweet.date = datetime.datetime.fromtimestamp(dateSec)
            tweet.formatted_date = datetime.datetime.fromtimestamp(dateSec).strftime("%a %b %d %X +0000 %Y")
            tweet.hashtags = " ".join(re.compile('(#\\w*)').findall(txt1))
            tweet.geo = geo

            counter += 1
            resultsAux.append(tweet)
            print(month + " download {}".format(counter))
            if receiveBuffer and len(resultsAux) >= bufferLength:
                output_counter += len(resultsAux)
                receiveBuffer(resultsAux)
                resultsAux = []
                print(month + " {} Tweets saved on file..".format(output_counter))

            if tweetCriteria.maxTweets > 0 and counter >= tweetCriteria.maxTweets:
                active = False
                break

        if receiveBuffer and len(resultsAux) > 0:
            output_counter += len(resultsAux)
            receiveBuffer(resultsAux)
            print(month + " {} Tweets saved on file..".format(output_counter))

        print("We already download {0} tweets on" + month + ", and the ID of last Tweets is {1}".format(counter, tweet.id))
        return True

    # @staticmethod
    def generateUrl(self, tweetCriteria):

        urlGetData = ''
        if hasattr(tweetCriteria, 'username'):
            urlGetData += ' from:' + tweetCriteria.username

        if hasattr(tweetCriteria, 'since'):
            urlGetData += ' since:' + tweetCriteria.since

        if hasattr(tweetCriteria, 'until'):
            urlGetData += ' until:' + tweetCriteria.until

        if hasattr(tweetCriteria, 'querySearch'):
            urlGetData += ' ' + tweetCriteria.querySearch

        if hasattr(tweetCriteria, 'lang'):
            urlLang = 'lang=' + tweetCriteria.lang + '&'
        else:
            urlLang = ''

        return [urlGetData, urlLang]

    # @staticmethod
    def getJsonReponse(self, urlInfo, refreshCursor, cookieJar, month):
        url = "https://twitter.com/i/search/timeline?f=realtime&q=%s&src=typd&%smax_position=%s"

        url = url % (urllib.parse.quote(urlInfo[0]), urlInfo[1], refreshCursor)

        headers = [
            ('Host', "twitter.com"),
            ('User-Agent', "Mozilla/5.0 (Windows NT 6.1; Win64; x64)"),
            ('Accept', "application/json, text/javascript, */*; q=0.01"),
            ('Accept-Language', "de,en-US;q=0.7,en;q=0.3"),
            ('X-Requested-With', "XMLHttpRequest"),
            ('Referer', url),
            ('Connection', "keep-alive")
        ]

        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookieJar))
        opener.addheaders = headers

        try:
            time.sleep(1)
            response = opener.open(url)
            jsonResponse = response.read()
        except urllib.error.HTTPError as inf:

            f = open(month + '.txt', '+w')
            f.write(refreshCursor)
            f.flush()
            f.close()
            print(inf.reason)
            sys.exit()
        except Exception as inf:
            print(month + " Twitter weird response. Try to see on browser: ", url)
            print(month + " Twitter weird response. Try to see on browser: https://twitter.com/search?q=%s&src=typd" % urllib.parse.quote(urlInfo[0]))
            print(month + " Unexpected error:", sys.exc_info()[0])
            f = open(month + '.txt', '+w')
            f.write(refreshCursor)
            f.flush()
            f.close()
            sys.exit()
        else:
            dataJson = json.loads(jsonResponse.decode())
            return dataJson