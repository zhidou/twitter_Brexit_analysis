from multiprocessing import Pool
from scrape import scrape
import os, json, time

def load_data(data):
    criteria = {}
    currentPath = os.getcwd()
    for month in data:
        fname = month + '.txt'
        if os.path.isfile(os.path.join(currentPath, fname)):
            with open(fname, 'r') as f:
                ss = json.load(f)
                for keys in ss.keys():
                    criteria[keys] = ss[keys]
        else:
            print("No data exist!!")
            return {}
    return criteria




def main(data, resume=False):
    if type(data) == list and resume:
        data = load_data(data)

    retry = 0
    while True:
        try:
            if len(data) == 1:
                scrape(data[0])
            else:
                pool = Pool(processes=len(data))
                pool.map(scrape, data)
                pool.join()
                pool.close()
        except KeyboardInterrupt:
            choice = input("Do you want to continue? (Y/N)")
            if choice == 'N': break
            elif choice == 'Y':
                month = [x['month'] for x in data]
                load_data(month)
                pass
            else:
                print("exit main function")
                break
        except Exception as inst:
            print("Error happens!!!")
            if retry < 3:
                print("sleep 300s and retry!")
                time.sleep(300)
                month = [x['month'] for x in data]
                load_data(month)
                retry += 1
                pass
            else:
                print("fail!!!!")
                break
        else:
            print("We successfully scrape {} month data!!!".format(len(data)))


if __name__ == '__main__':
    dic1={}
    dic2={}
    dic3={}
    dic1['querysearch'] = 'Brexit'
    dic1['since'] = '2016-05-01'
    dic1['until'] = '2016-06-01'
    dic1['topTweets'] = True
    dic1['lang'] = 'en'
    dic1['refreshCursor'] = ''
    dic1['month'] = 'May'

    dic2['querysearch'] = 'Brexit'
    dic2['since'] = '2016-07-01'
    dic2['until'] = '2016-08-01'
    dic2['topTweets'] = True
    dic2['lang'] = 'en'
    dic2['refreshCursor'] = ''
    dic2['month'] = 'July'
    # dic3['querysearch'] = 'Brexit'
    # dic3['since'] = '2016-08-01'
    # dic3['until'] = '2016-09-01'
    # dic3['topTweets'] = True
    # dic3['lang'] = 'en'
    # dic3['refreshCursor'] = ''

    data1 = [dic1]
    data2 = [dic1, dic2]
    beginT = time.time()
    main(data1)
    print("Tollay running time: {}".format(time.time() - beginT))
