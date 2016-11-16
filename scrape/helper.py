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
            return {}
    return criteria