from multiprocessing import Pool
from scrape import scrape
from helper import load_data, interrupt_handler_main
import time

def main(data, resume=False):
    if type(data) == list and resume:
        data = load_data(data)
<<<<<<< HEAD
    pool=[]
=======

>>>>>>> e25ab4dfd22d84c16f64a5d50f3ddcb6152cbca1
    while True:
        try:
            if len(data) == 1:
                scrape(data[0])
            else:
                for x in data: x['pid'] = -1
                pool = Pool(processes=len(data))
                pool.map(scrape, data)
                pool.close()
                pool.join()
        except KeyboardInterrupt:
<<<<<<< HEAD
            if interrupt_handler_main(KeyboardInterrupt, pool):
=======
            if interrupt_handler_main(KeyboardInterrupt):
>>>>>>> e25ab4dfd22d84c16f64a5d50f3ddcb6152cbca1
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


if __name__ == '__main__':
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
<<<<<<< HEAD
    data2 = [dic1, dic2, dic3]
=======
    data2 = [dic1, dic2]
>>>>>>> e25ab4dfd22d84c16f64a5d50f3ddcb6152cbca1
    beginT = time.time()
    # data3 = ['May1', 'May2']
    main(data1)

    print("Tollay running time: {}".format(time.time() - beginT))
