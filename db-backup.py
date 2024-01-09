# -*- coding: utf-8 -*-
"""
===================================
Demo of DBSCAN clustering algorithm
===================================

Finds core samples of high density and expands clusters from them.

"""
print(__doc__)

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import itertools
from scipy import linalg
import matplotlib as mpl

df = pd.read_csv('tweets.csv', usecols=[1,2])
df = df.dropna(how='any',axis=0)
latlon = df.to_numpy()
db = DBSCAN(eps=1/100, min_samples=20).fit(latlon)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
plt.scatter(latlon[:, 0], latlon[:, 1], c=labels)
X = StandardScaler().fit_transform(latlon)
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, labels))

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
# maxscore = -1
# clusterflag= 0
# for e in np.arange(1/100,1, 1/100):
#     db = DBSCAN(eps=e, min_samples=20).fit(latlon)
#     labels = db.labels_
#     score = metrics.silhouette_score(latlon, labels)
#     print(score)
#     if(score>maxscore):
#         maxscore = score
#         epsilon = e
# print ("For epsilon = {}, silhouette score is {})".format(epsilon, maxscore))
# #############################################################################

# df = pd.read_csv('tweets.csv')
# df[df.columns[1]].fillna(0, inplace=True)
# df[df.columns[2]].fillna(0, inplace=True)
# df.isna().sum()
# df.shape
# df.reset_index(drop=True)
# latlon = np.array(list(zip(df[df.columns[1]], df[df.columns[2]])))


# #############################################################################
# #Compute DBSCAN
# db = DBSCAN(eps=1/1000, min_samples=20).fit(X)
# # #db = DBSCAN(eps=.1/6371, min_samples=20, algorithm='ball_tree', metric='haversine').fit(np.radians(latlon))
# # # db = DBSCAN(eps=0.1/6371, min_samples=70, algorithm='ball_tree', metric='haversine').fit(latlon)
# # df['clusters'] = db.labels_
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_

# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)

# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(labels_true, labels,
#                                            average_method='arithmetic'))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, labels))

# #############################################################################
#Plot result

# # Black removed and is used for noise instead.
# unique_labels = set(labels)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]

#     class_member_mask = (labels == k)

#     xy = X[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=14)

#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=6)

# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()

