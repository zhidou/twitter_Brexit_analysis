
# coding: utf-8

# In[1]:

import json, csv, colorsys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
get_ipython().magic('matplotlib inline')
# np.set_printoptions(threshold=np.nan)


# In[2]:

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


# In[3]:

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


# In[17]:

plt.figure(figsize=(15,5))
plt.xticks(range(40))
plt.grid()
dummy = plt.plot(range(1,len(S1)+1),S1, '-o')
plt.ylabel('Singular Value',size=20)
dummy = plt.xlabel('Dimension',size=20)


# In[18]:

plt.figure(figsize=(15,5))
plt.xticks(range(40))
plt.grid()
dummy = plt.plot(range(1,len(S2)+1),S2, '-o')
plt.ylabel('Singular Value',size=20)
dummy = plt.xlabel('Dimension',size=20)


# In[19]:

plt.figure(figsize=(15,5))
plt.xticks(range(40))
plt.grid()
dummy = plt.plot(range(1,len(S3)+1),S3, '-o')
plt.ylabel('Singular Value',size=20)
dummy = plt.xlabel('Dimension',size=20)


# In[20]:

plt.figure(figsize=(15,5))
plt.xticks(range(40))
plt.grid()
dummy = plt.plot(range(1,len(S4)+1),S4, '-o')
plt.ylabel('Singular Value',size=20)
dummy = plt.xlabel('Dimension',size=20)


# In[21]:

terms1 = np.array(Vectorize1.get_feature_names())
tops1 = []
for i in range(8):
    top1 = VT1[i].argsort()[::-1]
    topterms1 = [terms1[top1[f]] for f in range(50)]
    print(i, topterms1)
    tops1.append(set(topterms1))


# In[22]:

terms2 = np.array(Vectorize2.get_feature_names())
tops2 = []
for i in range(8):
    top2 = VT2[i].argsort()[::-1]
    topterms2 = [terms2[top2[f]] for f in range(50)]
    print(i, topterms2)
    tops2.append(set(topterms2))


# In[23]:

terms3 = np.array(Vectorize3.get_feature_names())
tops3 = []
for i in range(8):
    top3 = VT3[i].argsort()[::-1]
    topterms3 = [terms3[top3[f]] for f in range(50)]
    print(i, topterms3)
    tops3.append(set(topterms3))


# In[24]:

terms4 = np.array(Vectorize4.get_feature_names())
tops4 = []
for i in range(8):
    top4 = VT4[i].argsort()[::-1]
    topterms4 = [terms4[top4[f]] for f in range(50)]
    print(i, topterms4)
    tops4.append(set(topterms4))


# ## Top Five Tags for Remain

# 0. Economy/Trade
# 1. Jobs/Market/Risk
# 2. Law/Govern
# 3. Security
# 4. Nation/Leader

# ## Top Five Tags for Leave

# 1. Freedom/Sovereignty/Great
# 2. Tax
# 3. Economy/Trade
# 4. Migrants
# 5. Lie/Brussels

# In[25]:

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
    print(k)


# In[23]:

dummy = plt.figure(figsize=(12,5))
dummy = plt.plot(range(2,50),ss, '-o')
plt.xticks(range(50))
plt.grid()
plt.ylabel('Silhouette Score',size=20)
plt.xlabel('No of Prin Comps',size=20)


# In[24]:

vectorsk = Xk[:,:15]
kmeans = KMeans(n_clusters=15, init='k-means++', max_iter=100, n_init=10, random_state=0)
kmeans.fit_predict(vectorsk)
labelsk = kmeans.labels_


# In[25]:

center = kmeans.cluster_centers_


# In[26]:

arg = center.argsort()


# In[27]:

df_cluster = pd.DataFrame(np.array(range(15)),columns=['cluster'])
df_cluster['major_component'] = arg[:,-1]
df_cluster.index = df_cluster['major_component']
df_cluster = pd.merge(df_cluster, categories, left_index=True, right_index=True, how='outer')
df_cluster = df_cluster.dropna()


# In[28]:

df_cluster.index = df_cluster['cluster']
df_review['cluster'] = labelsk
df_review.index = df_review['cluster']
result = pd.merge(df_review, df_cluster, left_index=True, right_index=True, how='outer')


# In[29]:

df.index = df['business_id']
result.index = result['business_id']
df = pd.merge(result, df, left_index=True, right_index=True, how='outer')
df['Index'] = np.array(range(len(df)))
df['business_id'] = df['business_id_x']
df.drop('business_id_y',axis = 1, inplace=True)
df.drop('business_id_x',axis = 1, inplace=True)
df.drop('cluster_y',axis = 1, inplace=True)
df.drop('cluster_x',axis = 1, inplace=True)
df.drop('Unnamed: 0',axis = 1, inplace=True)


# Find clusters using the 3 different techniques we discussed in class: k-means++, hierarchical, and GMM. Visualize the clusters by plotting the longitude/latitude of the restaurants in a scatter plot and label each cluster. 
# 
# Note that to label each cluster, you will need to think about how to extract labels from the LSA results.
# **(25 pts)**

# ## Hierarchical Clustering

# In[30]:

feature = set_feature(df, 0.7)


# In[31]:

plt.figure(figsize=(15,5))
Z = hr.linkage(feature, method='ward', metric='euclidean')
T = hr.dendrogram(Z,color_threshold=0.4, leaf_font_size=4)


# In[32]:

last = Z[-50:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.figure(figsize=(12,5))
_ = plt.plot(idxs, last_rev)
plt.xlabel('Number of clusters')
plt.xticks(range(50))
plt.grid(True)
dummy = plt.ylabel('Longest Distance with in each clusters')


# In[33]:

hnum = 15
hcluster = hr.fcluster(Z, hnum ,'maxclust') - 1


# In[34]:

df['hr'] = hcluster
df.index = df['hr']
# print the size of each cluster
for i in range(hnum):
    print("Cluster {} size : {}".format(i, len(df.ix[i])))


# In[35]:

# print the label of each cluster
htag = []
Vectorizer = TfidfVectorizer()
for i in range(hnum):
    hcategories = Vectorizer.fit_transform(df.ix[i]['categories'].values).toarray()
    hTopIndex = hcategories.sum(axis = 0).argsort()[-2:]
    hTop1Ca = np.array(Vectorizer.get_feature_names())[hTopIndex]
    if len(hTop1Ca) == 2:
        temp = hTop1Ca[0]+'/'+hTop1Ca[1]
    else: temp = hTop1Ca[0]
    htag.append(temp)
    print('Cluster {} :'.format(i + 1)+ temp)


# In[36]:

draw_map(hnum, 'hr', df, hcluster, htag)


# ## Kmeas

# In[37]:

feature = set_feature(df, 0.5)


# In[38]:

# draw the graph of error and adjusted rand index
error = np.zeros(51)
s = np.zeros(51)
for k in range(1, 51):
    kmeans = KMeans(init = 'k-means++', n_clusters = k, n_init = 10)
    kmeans.fit_predict(feature)
    error[k] = kmeans.inertia_
    if k >= 2: s[k] = metrics.silhouette_score(feature, kmeans.labels_, metric='cosine')


# In[39]:

plt.figure(figsize=(15,5))
plt.plot(range(2,len(s)),s[2:])
plt.xlabel('Number of clusters')
plt.grid(True)
plt.xticks(range(50))
dummpy = plt.ylabel('Adjusted Rand Index')


# In[40]:

plt.figure(figsize=(15,5))
plt.plot(range(1, len(error)), error[1:])
plt.xlabel('Number of clusters')
plt.grid(True)
plt.xticks(range(50))
dummpy = plt.ylabel('Error')


# In[41]:

knum = 15
kmeans = KMeans(n_clusters = knum, init = 'k-means++', max_iter = 100, n_init = 10)
kcluster = kmeans.fit_predict(feature)


# In[42]:

# print the size of each Cluster
df['kmeans'] = kcluster
df.index = df['kmeans']
for i in range(knum):
    print('Cluster {} : size {}'.format(i, len(df.ix[i]['business_id'])))


# In[43]:

# print the first tag of each cluster and the last tag
df.index = df['Index']
kclasses = [[] for i in range(knum)]
kindeces = [[] for i in range(knum)]
kmeanTopTag = []
for i, e in enumerate(kcluster):
    kclasses[e].append(feature[i])
    kindeces[e].append(i)
ktag = []
print('Categories of {} cluster by Kmeans'.format(knum))
for k in range(knum):
    cdis = euclidean_distances(kclasses[k], [kmeans.cluster_centers_[k]]).T.argsort()
    top1 = cdis[0][0]
    ktag.append(df.ix[kindeces[k][top1]]['categories'])
    print('Cluster {}: '.format(k + 1)+ df.ix[kindeces[k][top1]]['categories'])


# In[44]:

draw_map(knum, 'kmeans', df, kcluster, ktag)


# ## GMM

# In[45]:

feature = set_feature(df, 0.8)


# In[46]:

# to estimate GMM model
error = np.zeros(30)
for k in range(1, 30):
    gmm = mixture.GaussianMixture(n_components=k, covariance_type='full')
    gmm.fit(feature)
    gmm.predict(feature)
    error[k] = gmm.score(feature)


# In[47]:

# plot the average likelyhood to select the most 
plt.figure(figsize=(15,5))
plt.plot(range(1, len(error)), error[1:])
plt.xlabel('Number of clusters')
plt.grid(True)
plt.xticks(range(30))
dummpy = plt.ylabel('Likelyhood')


# In[48]:

# Do GMM classification and plot
gnum = 12
gmm = mixture.GaussianMixture(n_components=gnum, covariance_type='full')
gmm.fit(feature)
gclusters = gmm.predict(feature)
df['GMM']= gclusters


# In[49]:

# get the probability of each point in each class and get the largest probability
pro = gmm.predict_proba(feature)
pros = [i.max() for i in pro]
# build a data frame contain the cluster this point belonging to, and its probability, and its index in the origin table
df_gmm = pd.DataFrame(data=gclusters, columns = ['cluster'])
df_gmm['Pro'] = pros
df_gmm['index'] = np.array(range(len(feature)))
# make cluster as the index of data frame
df_gmm.index = df_gmm['cluster']
# print the size of each cluster
print('Categories of {} cluster by GMM'.format(gnum))
for k in range(gnum):
    print('Cluster {} size: {}'.format(k + 1, len(df_gmm.ix[k])))


# In[50]:

# for each cluster sort the its point in the probability, take the point has most prosibble belonging this 
# cluster, its label as the label for this cluster
df.index = df['Index']
gtag = []
for k in range(gnum):
    first30 = df_gmm.ix[k].sort_values(['Pro'], ascending = False)['index'][0:30].values
    last30 = df_gmm.ix[k].sort_values(['Pro'], ascending = False)['index'][-30: ].values
    gtag.append(df.ix[first30[0]]['categories'])
    print('Cluster {}: '.format(k + 1)+ df.ix[first30[0]]['categories'])


# In[51]:

draw_map(gnum, 'GMM', df, gclusters, gtag)


# Compare your clusters with the results you obtained in Part 1. Use cluster comparison metrics, and also comment on which clustering appears (from your inspection of the clusters) to be more informative, and why. **(15 pts)**

# ## Compare with the clusters with the results get from part 1

# The result we get from part 2 mostly are most like but still different from the result we get from part 1, I will analyze the reason.
# First, although more than half catigories we get from reviews are similar with the true catigories, there are still difference. This is reasonalbe. Since the reviews of customers on a restruant are not totally about the "catigories" about these restraunt! Of course, the reviews may contain these information, but still most reviews are about the "service", "teste", "paid" and so on. And also the reviews may on some specific material, like "pork", "chicken","shrimp" and so on. We may have food with "chicken" in Chinese restraunt, Japanese restraunt and also american restraunt! Sometimes we could still get catigories from these material, like "shrimp", "seefood" always related with "Sushi", but we are not always that lucky.
# Second, because of the categories are not totally the same, our clasification is not totally the same! For the high frequency catigories like "Pizza", "Maxican" and so on, the cluster of these restraunts could be almost the same, from the graph. But still because we have far less categories some detail we cannot get from part2.
# Below we will compare part1 with part1, part2 with part2 and part1 with part2.

# In[52]:

# load data we need for part1
df_part1 = pd.read_csv("restaurant_part1.csv")
hscaling = 0.3
kscaling = 0.3
gscaling = 0.01


# In[53]:

df_part1.rename(index=str, columns={"Latitude":"latitude","Longtitude":"longitude","Categories": "categories"}, inplace=True)


# In[54]:

feature = set_feature(df_part1, hscaling)
Z = hr.linkage(feature, method='ward', metric='euclidean')
hcluster_part1 = hr.fcluster(Z, 25 ,'maxclust') - 1


# In[55]:

feature = set_feature(df_part1, kscaling)
knum_part1 = 30
kmeans_part1 = KMeans(n_clusters = knum_part1, init = 'k-means++', max_iter = 100, n_init = 10)
kcluster_part1 = kmeans.fit_predict(feature)


# In[56]:

feature = set_feature(df_part1, gscaling)
gnum_part1 = 26
gmm_part1 = mixture.GaussianMixture(n_components=gnum_part1, covariance_type='full')
gmm_part1.fit(feature)
gcluster_part1 = gmm_part1.predict(feature)


# ## part2 with part2 

# ### Hierarchical Clustering

# In[57]:

feature = set_feature(df, 0.7)


# In[58]:

ri = []
ss = []
Z = hr.linkage(feature, method='ward', metric='euclidean')
for k in range(1, 50):
    hcluster = hr.fcluster(Z, k ,'maxclust') - 1
    ri.append(metrics.adjusted_rand_score(hcluster[0:len(hcluster_part1)],hcluster_part1))
    if k >= 2: ss.append(metrics.silhouette_score(feature,hcluster,metric='euclidean'))


# In[59]:

draw_Adjusted(ss)


# In[60]:

draw_Rand(ri)


# ### Kmeans

# In[61]:

error = np.zeros(51)
ri = []
ss = []
feature = set_feature(df, 0.5)
for k in range(1,50):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=10, random_state=0)
    kmeans.fit_predict(feature)
    labelsk = kmeans.labels_
    error[k] = kmeans.inertia_
    ri.append(metrics.adjusted_rand_score(labelsk[0:len(kcluster_part1)],kcluster_part1))
    if k >= 2: ss.append(metrics.silhouette_score(feature,kmeans.labels_,metric='euclidean'))


# In[62]:

plt.figure(figsize=(15,5))
plt.plot(range(2,len(ss)),ss[2:])
plt.xlabel('Number of clusters',size=20)
plt.grid(True)
plt.xticks(range(50))
dummpy = plt.ylabel('Adjusted Rand Index',size=20)


# In[63]:

plt.figure(figsize=(15,5))
plt.plot(range(1,50),ri)
plt.xticks(range(50))
plt.grid()
plt.ylabel('Rand Score',size=20)
dummy = plt.xlabel('No of Prin Comps',size=20)


# In[64]:

draw_Error(error)


# In[65]:

print('Error at clusters = 15:{}'.format(error[15]))


# ### GMM

# In[66]:

error = np.zeros(51)
ri = []
ss = []
feature = set_feature(df, 0.8)
for k in range(1,50):
    gmm = mixture.GaussianMixture(n_components=k, covariance_type='full')
    gmm.fit(feature)
    gclusters = gmm.predict(feature)
    ri.append(metrics.adjusted_rand_score(gclusters[0:len(gcluster_part1)],gcluster_part1))
    if k >= 2: ss.append(metrics.silhouette_score(feature,gclusters,metric='euclidean'))


# In[67]:

draw_Adjusted(ss)


# In[68]:

draw_Rand(ri)


# ## Par1 with Part1

# ### Hierarchical Clustering

# In[69]:

feature = set_feature(df_part1, hscaling)


# In[70]:

ss_h1 = []
Z = hr.linkage(feature, method='ward', metric='euclidean')
for k in range(1, 50):
    hcluster = hr.fcluster(Z, k ,'maxclust') - 1
    if k >= 2: ss_h1.append(metrics.silhouette_score(feature,hcluster,metric='euclidean'))


# In[71]:

draw_Adjusted(ss_h1)


# ### Kmeans

# In[72]:

error_k1 = np.zeros(51)
ss_k1 = []
feature = set_feature(df_part1, 0.3)
for k in range(1,50):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=10, random_state=0)
    kmeans.fit_predict(feature)
    labelsk = kmeans.labels_
    error_k1[k] = kmeans.inertia_
    if k >= 2: ss_k1.append(metrics.silhouette_score(feature,kmeans.labels_,metric='euclidean'))


# In[73]:

draw_Adjusted(ss_k1)


# In[74]:

draw_Error(error_k1)


# In[75]:

print('Error at clusters 30: {}'.format(error_k1[30]))


# ### GMM

# In[78]:

ss_g1 = []
feature = set_feature(df_part1, 0.01)
for k in range(1,50):
    gmm = mixture.GaussianMixture(n_components=k, covariance_type='full')
    gmm.fit(feature)
    gclusters = gmm.predict(feature)
    if k >= 2: ss_g1.append(metrics.silhouette_score(feature,gclusters,metric='euclidean'))


# In[79]:

draw_Adjusted(ss_g1)


# ## Part2 with Part1

# In "Part2 with part2", we set the best result we get from "part1" as the groud true, compared with result we get from part2 to see the variance between these two. From the graphs "rand Score" in "part2 with part2" we could see the result we get from part2 is not that difference with the result we get from part1, the variation is between (-0.01, 0.01). But we could also see, generally the in part2, we could get a better result in less cluster. We could get a better Ajusted Rand Index Score with less clusters and fastly for each method. And also more robust. For example Hierarchical Clustering in part2 will converge to round 0.4 after 10 cluster, but in part1, it goes increasing but, after 49 clusters it just have 0.29 scores. And also for Kmeans we get error 5700 for just having 15 cluster in part2, but we should have 30 cluster to reduce error to 5000 part1.
# 
# Thus generally, the cluster we get from part2 is more robust and clear. This may be we have far less categories than we have in part 1. This is a good thing, not also a bad thing. For good thing, we get robust model. For bad, the model may not tell us the detail.

# ## Compare methods

# Comparet these 3 methods. We could find GMM could get the highest Ajusted Rand Index Score fast than other two, but with the number of clusters increasing, the score of GMM is decreasing but others two will increase. But, Ajusted Rand Index cannot equal to informative. From Both Part1 and Part2, we could get more better information from Kmeans, not just because Kmeans is more easier to implement and with less parameters, but the most important is, it depends on distance which is fit for most problem. GMM is powerful, but not every problem chould fit into Gaussian model. And For Hierarchical Clustering, it is better to see the clustering directly, but it is hard to clearly say how many cluster and what is the cluster. And for Hierarchical Clustering it is hard to say the extent a point belong to a certain cluster, which we could do this in Kmeans and GMM.
# 
# Generally I think, Hierarchical Clustering could give we a teast of how many cluster we should chose at the beginning, which is realy imporant information. And then we could use GMM to train this data, if we get a good clustering, then we are lucky. These data are fit into Gaussian model. But if it is not good, then we should try Kmeans. Of course there is limitation of Kmeans, for example, if the restraunts have same categories but a little bit far, than it will not be contained. But gernerally Kmeans could give us a big picture of the cluster.
