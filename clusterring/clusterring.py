import json, csv, colorsys
import pandas as pd
import numpy as np
import scipy.sparse.linalg as linalg
import scipy.cluster.hierarchy as hr
import sklearn.metrics as metrics
import sklearn.manifold as manifold
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.feature_extraction import text 
from sklearn.utils.extmath import randomized_svd
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

# set the colors we use to plot
def set_color(num):
    ah = 1.0 / num
    l = 0.6
    s = 0.8
    
    colors = [list(colorsys.hls_to_rgb(i * ah, l ,s)) for i in range(num)]
    colors = np.array(colors)
    return colors


# In[4]:

# The function we use to draw figure
def draw_map(num, tag, df, cluster, tags = []):
    fig = plt.figure(1)
    
    df[tag] = cluster
    df.index = df[tag]
    
    top_lat = 36.310359
    bottom_lat = 35.980255
    left_long = -115.357904
    right_long = -114.949179
    
    delt_lat = (top_lat - bottom_lat) / 1200.
    delt_long = (right_long - left_long) / 1200.
    
    colors = set_color(num)
    plt.figure(figsize=(10,10))
    im =plt.imread('vegas.png')
    implot = plt.imshow(im, zorder=0, extent=[0, 1200, 0, 1200])
    ax = plt.subplot(111)
    for i in range(num):
        pos = df.ix[i][['latitude', 'longitude']].values
        plt.scatter((pos[:, 0] - bottom_lat) / delt_lat, (pos[:, 1] - left_long) / delt_long, color=colors[i], alpha=0.5)
        mean = pos.mean(axis=0)
    if len(tags) != 0:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(tags, scatterpoints=1, prop={'size':6},loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.ylim([0,1200])
    plt.xlim([0,1200])
    plt.xticks(())
    plt.yticks(())
    df.index = df['Index']


# In[5]:

# combine coordinate with categories and standardize the vector
def set_feature(df, scaling = 0):
    Vectorizer = TfidfVectorizer()
    Vcategories = Vectorizer.fit_transform(df['categories'].values)
    
    coordinate = df[['latitude','longitude']]
    categories = Vcategories.toarray()
    
    feature = np.concatenate((coordinate, categories),axis = 1)
    norm = StandardScaler()
    feature = norm.fit_transform(feature)
    # because the categories has more demonsion than coordinate, should we should scaling the vector
    feature[:, 2 : ] *= scaling
    return feature


# In[6]:

def draw_Adjusted(ss):
    plt.figure(figsize=(15,5))
    plt.plot(range(2,len(ss)),ss[2:])
    plt.xlabel('Number of clusters',size=20)
    plt.grid(True)
    plt.xticks(range(50))
    dummpy = plt.ylabel('Adjusted Rand Index',size=20)


# In[7]:

def draw_Rand(ri):
    plt.figure(figsize=(15,5))
    plt.plot(range(1,50),ri)
    plt.xticks(range(50))
    plt.grid()
    plt.ylabel('Rand Score',size=20)
    dummy = plt.xlabel('No of Prin Comps',size=20)


# In[8]:

def draw_Error(error):
    plt.figure(figsize=(15,5))
    plt.plot(range(1, len(error)-1), error[1:-1])
    plt.xlabel('Number of clusters',size=20)
    plt.grid(True)
    plt.xticks(range(50))
    dummpy = plt.ylabel('Error',size=20)


# In[9]:

df_remain = pd.read_csv('remain.csv')
df_leave = pd.read_csv('leave.csv')


# In[10]:

stemmer = PorterStemmer()
stemmed_data = [" ".join(stemmer.stem(word)  for sent in sent_tokenize(message) for word in word_tokenize(sent)) for message in df_remain['text']]
df_remain['text2'] = stemmed_data
stemmed_data = [" ".join(stemmer.stem(word)  for sent in sent_tokenize(message) for word in word_tokenize(sent)) for message in df_leave['text']]
df_leave['text_2'] = stemmed_data


# In[11]:

Vectorize1 = TfidfVectorizer(min_df = 0.006, max_df = 0.6, stop_words = 'english')
dtm_remain = Vectorize1.fit_transform(df_remain['text'].values)


# In[12]:

Vectorize2 = TfidfVectorizer(min_df = 0.006, max_df = 0.6, stop_words = 'english')
dtm_remain_stem = Vectorize2.fit_transform(df_remain['text2'].values)


# In[13]:

Vectorize3 = TfidfVectorizer(min_df = 0.006, max_df = 0.6, stop_words = 'english')
dtm_leave = Vectorize3.fit_transform(df_leave['text'].values)


# In[14]:

Vectorize4 = TfidfVectorizer(min_df = 0.006, max_df = 0.6, stop_words = 'english')
dtm_leave_stem = Vectorize4.fit_transform(df_leave['text_2'].values)


# In[15]:

dtm_dense_remain = dtm_remain.todense()
centered_dtm_remain = dtm_dense_remain - np.mean(dtm_dense_remain, axis = 0)

dtm_dense_remain_stem = dtm_remain_stem.todense()
centered_dtm_remain_stem = dtm_dense_remain_stem - np.mean(dtm_dense_remain_stem, axis = 0)

dtm_dense_leave = dtm_leave.todense()
centered_dtm_leave = dtm_dense_leave - np.mean(dtm_dense_leave, axis = 0)

dtm_dense_leave_stem = dtm_leave_stem.todense()
centered_dtm_leave_stem = dtm_dense_leave_stem - np.mean(dtm_dense_leave_stem, axis = 0)


# In[16]:

U1, S1, VT1 = randomized_svd(centered_dtm_remain, n_components=40, n_iter=10)
U2, S2, VT2 = randomized_svd(centered_dtm_remain_stem, n_components=40, n_iter=10)
U3, S3, VT3 = randomized_svd(centered_dtm_leave, n_components=40, n_iter=10)
U4, S4, VT4 = randomized_svd(centered_dtm_leave_stem, n_components=40, n_iter=10)


Xk1 = U1 @ np.diag(S1)


# In[26]:

Xk1.shape


# In[ ]:

ss = []
for k in range(1,30):
    vectorsk1 = Xk1[:,:k]
    kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=10, random_state=0)
    kmeans.fit_predict(vectorsk1)
    labelsk1 = kmeans.labels_
    if k >= 2: ss.append(metrics.silhouette_score(vectorsk1,kmeans.labels_,metric='euclidean'))

df_error = pd.DataFrame(ss, columns=['error'])
df_error.to_csv('error.csv')


