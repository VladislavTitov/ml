import numpy as np
import random
from matplotlib import pyplot as plt
from math import sqrt
import gui

#A1 = np.array([[np.random.uniform(0, 20), np.random.uniform(0, 20)] for k in range(8)])
#A2 = np.array([[1,1], [4,6], [3,5], [7,7]])

#def dist(list1, list2):
#    return sum((i - j)**2 for i, j in zip(list1, list2))

#dist = np.array([[dist(i, j) for i in A2] for j in A1])
# print(dist)

#m = 1.1
#u = (1 / dist)**(1/(m-1))

#um = (u / u.sum(axis=1)[:, None])

#C = (um.T).dot(A1) / um.sum(axis=0)[:, None]

#print(C)




class CMeans:
    def __init__(self, dataset, n_clusters=3, m=2, cut_param=.9, dist='euclidian'):
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.m = m
        self.cut_param = cut_param
        self.max_n_iter = 10
        self.tolerance = .01
        self.dists = np.zeros((self.dataset.shape[0], self.n_clusters))
        self.dist = dist
        self.centroids = np.zeros((self.n_clusters, self.dataset.shape[1]))
        self.u = np.array([[np.random.uniform(0,1) for i in range(self.n_clusters)] for j in range(self.dataset.shape[0])])
        
        #self.choose_centroids() # np.array([self.dataset[k] for k in range(self.n_clusters)], dtype='f')
        print(self.centroids)

    def choose_centroids(self):
        return np.array([self.dataset[k] for k in random.sample(range(len(self.dataset)), self.n_clusters)])

    def get_dist(self, list1, list2):
        if self.dist == 'euclidian':
            return self.get_euclidian_dist(list1, list2)
        elif self.dist == 'plane': 
            return self.get_abs_dist(list1, list2)
        elif self.dist == 'square':
            return self.get_dist2(list1, list2)
        else:
            raise ValueError('Not correct distance is specified!')

    def get_dist2(self, list1, list2):
        return sum((i - j)**2 for i,j in zip(list1, list2))

    def get_euclidian_dist(self, list1, list2):
        return sqrt(sum((i - j)**2 for i,j in zip(list1, list2)))

    def get_abs_dist(self, list1, list2):
        return sum(abs(i - j) for i,j in zip(list1, list2))

    def distribute_data(self):
        self.dists = np.array([[self.get_dist(i, j) for i in self.centroids] for j in self.dataset])
        self.u = (1 / self.dists)**(1/(self.m-1))
        self.u = (self.u / self.u.sum(axis=1)[:, None])

    def recalculate_centroids(self):
        self.centroids = (self.u.T).dot(self.dataset) / self.u.sum(axis=0)[:, None]

    def fit(self):
        iter = 1
        while iter < self.max_n_iter:
            prev_centroids = np.copy(self.centroids)
            self.recalculate_centroids()
            self.distribute_data()
            if max([self.get_dist(i, k) for i, k in zip(self.centroids, prev_centroids)]) < self.tolerance:
                break
            iter += 1

    def get_labels(self):
        return [j if i>self.cut_param else 0 for i, j in zip(self.u.max(axis=1), [m+1 for m in self.u.argmax(axis=1)])]


    def show(self):
        if self.n_clusters > 10 or self.dataset.ndim != 2:
            return
        cmap = plt.cm.get_cmap('hsv', len(self.get_labels()))
        plt.figure()
        for i, (X, Y) in zip(self.get_labels(), self.dataset):
            plt.scatter(X, Y, c=cmap(i))
        #plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='x', s=100, c='black')
        plt.show()

#dataset = np.array([[np.random.uniform(0, 20), np.random.uniform(0, 20)] for k in range(8)])
dataset = gui.dataset
test = CMeans(dataset, 2, 1.1, 0.9)
test.fit()
test.show()
