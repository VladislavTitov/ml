import numpy as np
import random
from matplotlib import pyplot as plt
from math import sqrt

l1 = np.array([[1,2], [3,4], [5,6], [6,7]])

class KMeans:
    def __init__(self, dataset, n_clusters=3, dist='euclidian'):
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.max_n_iter = 10
        self.tolerance = .01
        self.fitted = False
        self.dist = dist
        self.labels = np.array([])
        self.centroids = self.choose_centroids() # np.array([self.dataset[k] for k in range(self.n_clusters)], dtype='f')
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
        self.labels = np.array([])
        for elem in self.dataset:
            dist2 = [self.get_dist(elem, center) for center in self.centroids]
            idx = dist2.index(min(dist2))
            self.labels = np.append(list(self.labels), idx).astype(int)

    def recalculate_centroids(self):
        for i in range(self.n_clusters):
            num = 0
            temp = np.zeros(self.dataset[0].shape)
            for k, label in enumerate(self.labels):
                if label == i:
                    temp = temp + self.dataset[k]
                    num  += 1

            self.centroids[i] = temp/num

    def fit(self):
        iter = 1
        while iter < self.max_n_iter:
            prev_centroids = np.copy(self.centroids)
            self.distribute_data()
            self.recalculate_centroids()
            if max([self.get_dist(i, k) for i, k in zip(self.centroids, prev_centroids)]) < self.tolerance:
                break
            iter += 1
        self.fitted = True

    def predict(self, data):
        if self.fitted:
            dist2 = [self.get_dist(data, center) for center in self.centroids]
            return dist2.index(min(dist2))

    def show(self):
        if self.n_clusters > 10 or self.dataset.ndim != 2:
            return
        cmap = plt.cm.get_cmap('hsv', len(self.labels))
        plt.figure()
        for i, (X, Y) in zip(self.labels, self.dataset):
            plt.scatter(X, Y, c=cmap(i))
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='x', s=100, c='black')
        plt.show()

test = KMeans(l1, 3)
test.fit()
print(test.predict([4,4]))
test.show()

