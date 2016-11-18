import os, json


def load_data(data):
    criteria = []
    currentPath = os.getcwd()
    for month in data:
        fname = month + '.txt'
        dic={}
        if os.path.isfile(os.path.join(currentPath, fname)):
            with open(fname, 'r') as f:
                ss = json.load(f)
                for keys in ss.keys():
                    dic[keys] = ss[keys]
                criteria.append(dic)
        else:
            print("No data exist!!")
            return []
    return criteria


def interrupt_handler_main(excep, pool=[]):
    if pool:
        pool.close()
        pool.join()
    if excep == KeyboardInterrupt:
        choice = input("Do you want to continue? (Y/N)")
        if choice == 'N': return False
        elif choice == 'Y': return True
    else: return False

def interruptHandler(error, tweetCriteria , refreshCursor, total_counter):
    if len(error.args) > 0:
        print(error.args[0])
    tweetCriteria.dic['refreshCursor'] = refreshCursor
    tweetCriteria.dic['num'] = total_counter
    with open(tweetCriteria.month + '.txt', '+w') as f:
        json.dump(tweetCriteria.dic, f)