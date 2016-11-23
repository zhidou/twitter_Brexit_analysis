import urllib.request, urllib.parse, urllib.error, urllib.request, urllib.error,urllib.parse,json,re,datetime,sys,http.cookiejar
from textblob import TextBlob
import got3 as got
from pyquery import PyQuery
import time
import numpy as np
from helper import interruptHandler

def generateUrl(tweetCriteria):

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


def getJsonReponse(urlInfo, refreshCursor, cookieJar, month):
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
    except Exception as inst:
        raise Exception(inst)
    else:
        dataJson = json.loads(jsonResponse.decode())
        return dataJson


def getTweets(tweetCriteria, receiveBuffer = None, bufferLength = 100):
    month = tweetCriteria.month
    refreshCursor = tweetCriteria.refreshCursor

    total_counter = tweetCriteria.num
    output_counter = 0
    resultsAux = []
    cookieJar = http.cookiejar.CookieJar()

    active = True
    urlInfo = generateUrl(tweetCriteria)
    try:
        while active:
            tweets = []
            for i in range(4):
                json = getJsonReponse(urlInfo, refreshCursor, cookieJar, month)
                if len(json['items_html'].strip()) == 0:
                    break
                refreshCursor = json['min_position']
                tweets_html = PyQuery(json['items_html'])('div.js-stream-tweet')
                tweets.append(tweets_html[np.random.randint(0, len(tweets_html))])

            if len(tweets) == 0:
                break

            tweetHTML = tweets[np.random.randint(0, len(tweets))]

            tweetPQ = PyQuery(tweetHTML)
            tweet = got.models.Tweet()

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

            total_counter += 1
            resultsAux.append(tweet)
            print(month + " download {}".format(total_counter))
            if receiveBuffer and len(resultsAux) >= bufferLength:
                output_counter += len(resultsAux)
                receiveBuffer(resultsAux)
                resultsAux = []
                print(month + " {} Tweets saved on file..".format(output_counter))

            if tweetCriteria.maxTweets > 0 and total_counter >= tweetCriteria.maxTweets:
                active = False
                break
    except KeyboardInterrupt as inst:
        interruptHandler(inst, tweetCriteria, refreshCursor, total_counter)
        raise KeyboardInterrupt
    except Exception as inst:
        interruptHandler(inst, tweetCriteria, refreshCursor, total_counter)
        raise Exception(inst)
    else:
        print("We successfully download {0} tweets on".format(total_counter) + month)
    finally:
        if receiveBuffer and len(resultsAux) > 0:
            output_counter += len(resultsAux)
            receiveBuffer(resultsAux)
    return

