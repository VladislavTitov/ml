from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


colors = np.array(['#377eb8', '#ff7f00', '#4daf4a'])

iris = load_iris()
X = iris.data
y = iris.target

kmeans = KMeans(n_clusters=3).fit(X)
pred = kmeans.fit_predict(X)

fig = plt.figure()
ax = Axes3D(fig)
plt.scatter(X[:, 0], X[:,1], X[:,2], c=colors[pred])
plt.show()
