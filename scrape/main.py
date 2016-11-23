from multiprocessing import Pool
from scrape import scrape
<<<<<<< HEAD
import os, json, time

def load_data(data):
    criteria = []
    currentPath = os.getcwd()
    for month in data:
        fname = month['month'] + '.txt'
        dic={}
        if os.path.isfile(os.path.join(currentPath, fname)):
            with open(fname, 'r') as f:
                ss = json.load(f)
                for keys in ss.keys():
                    dic[keys] = ss[keys]
                criteria.append(dic)
        else:
            print("No data exist!!")
            return {}
    return criteria



=======
from helper import load_data, interrupt_handler_main
import time
>>>>>>> zhidou

def main(data, resume=False):
    if type(data) == list and resume:
        data = load_data(data)
    pool=[]
    while True:
        try:
            if len(data) == 1:
                data[0]['pid'] = 0
                scrape(data[0])
            else:
                for x in data: x['pid'] = -1
                pool = Pool(processes=len(data))
                pool.map(scrape, data)
                pool.close()
                pool.join()
        except KeyboardInterrupt:
            if interrupt_handler_main(KeyboardInterrupt, pool):
                month = [x['month'] for x in data]
                data = load_data(month)
                pass
            else:
                print("KeyboardInterrupt!!")
                break
        except Exception as inst:
            interrupt_handler_main(inst, pool)
            print("Error!!")
            raise
        else:
            print("We successfully scrape {} month data!!!".format(len(data)))
            break


if __name__ == '__main__':
<<<<<<< HEAD
	dic1={}
	dic2={}
	dic3={}
	dic1['querysearch'] = 'Brexit'
	dic1['since'] = '2016-05-01'
	dic1['until'] = '2016-05-10'
	dic1['topTweets'] = True
	dic1['lang'] = 'en'
	dic1['refreshCursor'] = ''
	dic1['month'] = 'May1'

	dic2['querysearch'] = 'Brexit'
	dic2['since'] = '2016-05-11'
	dic2['until'] = '2016-05-20'
	dic2['topTweets'] = True
	dic2['lang'] = 'en'
	dic2['refreshCursor'] = ''
	dic2['month'] = 'May2'

	dic3['querysearch'] = 'Brexit'
	dic3['since'] = '2016-05-20'
	dic3['until'] = '2016-05-31'
	dic3['topTweets'] = True
	dic3['lang'] = 'en'
	dic3['refreshCursor'] = ''
	dic3['month'] = 'May3'
=======
    dic1={}
    dic2={}
    dic3={}
    dic1['querysearch'] = 'Brexit'
    dic1['since'] = '2016-07-01'
    dic1['until'] = '2016-07-11'
    dic1['topTweets'] = True
    dic1['lang'] = 'en'
    dic1['refreshCursor'] = ''
    dic1['month'] = 'July1'
    dic1['num'] = 0

    dic2['querysearch'] = 'Brexit'
    dic2['since'] = '2016-07-11'
    dic2['until'] = '2016-07-21'
    dic2['topTweets'] = True
    dic2['lang'] = 'en'
    dic2['refreshCursor'] = ''
    dic2['month'] = 'July2'
    dic2['num'] = 0

    dic3['querysearch'] = 'Brexit'
    dic3['since'] = '2016-07-21'
    dic3['until'] = '2016-08-01'
    dic3['topTweets'] = True
    dic3['lang'] = 'en'
    dic3['refreshCursor'] = ''
    dic3['month'] = 'July3'
    dic3['num'] = 0

    data1 = [dic1]
    data2 = [dic1, dic2, dic3]
    beginT = time.time()
    data3 = ['June3']
    main(data3, True)
>>>>>>> zhidou

	data1 = [dic1]
	data2 = [dic1, dic2]
	beginT = time.time()
	data3 = [dic1,dic2,dic3]
	main(data3, resume = True)

	print("Tollay running time: {}".format(time.time() - beginT))
