import urllib.request, urllib.parse, urllib.error,urllib.request,urllib.error,urllib.parse,json,re,datetime,sys,http.cookiejar
from textblob import TextBlob
from got3 import models
from pyquery import PyQuery

class TweetManager:
    def __init__(self):
        pass
    @staticmethod
    def getTweets(tweetCriteria, receiveBuffer = None, bufferLength = 100):
        emoticons_str = r"""
            (?:
                [:=;] # Eyes
                [oO\-]? # Nose (optional)
                [D\)\]\(\]/\\OpP] # Mouth
            )"""

        regex_str = [
            # emoticons_str,
            r'<[^>]+>',  # HTML tags
            r'(?:@[\w_]+)',  # @-mentions
            # r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
            r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
            # r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
            # r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
            # r'(?:[\w_]+)', # other words
            # r'(?:\S)' # anything else
        ]

        refreshCursor = ''

        counter = 0
        resultsAux = []
        cookieJar = http.cookiejar.CookieJar()

        active = True

        while active:
            json = TweetManager.getJsonReponse(tweetCriteria, refreshCursor, cookieJar)
            if len(json['items_html'].strip()) == 0:
                break

            refreshCursor = json['min_position']
            tweets = PyQuery(json['items_html'])('div.js-stream-tweet')

            if len(tweets) == 0:
                break

            for tweetHTML in tweets:
                tweetPQ = PyQuery(tweetHTML)
                tweet = models.Tweet()

                # lang = tweetPQ('p.js-tweet-text').attr('lang')
                # if lang != 'en':
                #     continue

                # usernameTweet = tweetPQ("span.username.js-action-profile-name b").text();
                txt = re.sub(r"\s+", " ", tweetPQ("p.js-tweet-text").text().replace('# ', '#').replace('@ ', '@'))


                txt = re.sub(r"http[s]?://.(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+", '', txt)
                txt = re.sub(r'(?:@[\w_]+)', '', txt)
                txt = re.sub(r'[\w]+/(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]\s))+', '', txt)
                txt = re.sub(r'(?:./[0-9]+)', '', txt)
                txt = txt.replace(',', '')
                txt = txt.replace('/ ','')
                txt1 = txt
                txt = re.sub(r" \#", '', txt)

                blob = TextBlob(txt)
                polarity, subjectivity = blob.sentiment

                # retweets = int(tweetPQ("span.ProfileTweet-action--retweet span.ProfileTweet-actionCount").attr("data-tweet-stat-count").replace(",", ""));
                # favorites = int(tweetPQ("span.ProfileTweet-action--favorite span.ProfileTweet-actionCount").attr("data-tweet-stat-count").replace(",", ""));
                dateSec = int(tweetPQ("small.time span.js-short-timestamp").attr("data-time"))
                id = tweetPQ.attr("data-tweet-id")

                usr_id = tweetPQ.attr('data-user-id')
                # permalink = tweetPQ.attr("data-permalink-path");
                # user_id = int(tweetPQ("a.js-user-profile-link").attr("data-user-id"))

                geo = ''
                geoSpan = tweetPQ('span.Tweet-geo')
                if len(geoSpan) > 0:
                    geo = geoSpan.attr('title')
                # urls = []
                # for link in tweetPQ("a"):
                # 	try:
                # 		urls.append((link.attrib["data-expanded-url"]))
                # 	except KeyError:
                # 		pass
                tweet.id = id
                # tweet.permalink = 'https://twitter.com' + permalink
                # tweet.username = usernameTweet

                tweet.user_id = usr_id
                tweet.wordnouns = (",").join(blob.noun_phrases)
                tweet.polarity = polarity
                tweet.subjectivity= subjectivity
                tweet.text = txt
                tweet.date = datetime.datetime.fromtimestamp(dateSec)
                tweet.formatted_date = datetime.datetime.fromtimestamp(dateSec).strftime("%a %b %d %X +0000 %Y")
                # tweet.retweets = retweets
                # tweet.favorites = favorites
                # tweet.mentions = " ".join(re.compile('(@\\w*)').findall(tweet.text))
                tweet.hashtags = " ".join(re.compile('(#\\w*)').findall(txt1))
                tweet.geo = geo
                # tweet.urls = ",".join(urls)
                # tweet.author_id = user_id

                counter += 1
                resultsAux.append(tweet)

                if receiveBuffer and len(resultsAux) >= bufferLength:
                    receiveBuffer(resultsAux)
                    resultsAux = []

                if tweetCriteria.maxTweets > 0 and counter >= tweetCriteria.maxTweets:
                    active = False
                    break

        if receiveBuffer and len(resultsAux) > 0:
            receiveBuffer(resultsAux)

        print("We already download {0} tweets, and the ID of last Tweets is {1}".format(counter, tweet.id))
        return True\

    @staticmethod
    def getJsonReponse(tweetCriteria, refreshCursor, cookieJar):
        url = "https://twitter.com/i/search/timeline?f=realtime&q=%s&src=typd&%smax_position=%s"

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
        url = url % (urllib.parse.quote(urlGetData), urlLang, refreshCursor)
        #print(url)

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
            response = opener.open(url)
            jsonResponse = response.read()
        except:
            print("Twitter weird response. Try to see on browser: ", url)
            print("Twitter weird response. Try to see on browser: https://twitter.com/search?q=%s&src=typd" % urllib.parse.quote(urlGetData))
            print("Unexpected error:", sys.exc_info()[0])
            sys.exit()

        dataJson = json.loads(jsonResponse.decode())

        return dataJson