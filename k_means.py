from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np


colors = np.array(['#377eb8', '#ff7f00', '#4daf4a'])

X_1, _ = make_blobs(n_samples = 300, random_state = 42)

inertia = []

for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(X_1)
    inertia.append(np.sqrt(kmeans.inertia_))

plt.plot(range(1, 8), inertia, marker='s')
plt.xlabel('$k$')
plt.ylabel('$J(C_k)$')


kmeans = KMeans(n_clusters=3).fit(X_1)
pred = kmeans.fit_predict(X_1)
centroids  = kmeans.cluster_centers_

plt.figure()
plt.scatter(X_1[:, 0], X_1[:,1], c=colors[pred])
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='black')
plt.show()
