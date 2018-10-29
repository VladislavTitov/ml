from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
import gui

dataset = gui.dataset
colors = np.array(['#000000', '#377eb8', '#ff7379', '#112345'])

class DB_SCAN():
    def __init__(self, dataset, eps=30, min_samples=2):
        self.dataset = dataset
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters = 0
        self.clusters = {0:[]}
        self.visited = set()
        self.clustered = set()
        self.fitted = False
        self.fit()
        
    def get_dist(self, list1, list2):
        return np.sqrt(sum((i-j)**2 for i, j in zip(list1, list2)))

    def get_region(self, data):
        return [list(q) for q in self.dataset if self.get_dist(data, q) < self.eps]

    def fit(self):
        for p in self.dataset:
            if tuple(p) in self.visited:
                continue
            self.visited.add(tuple(p))
            neighbours = self.get_region(p)
            if len(neighbours) < self.min_samples:
                self.clusters[0].append(list(p))
            else: 
                self.n_clusters += 1
                self.expand_cluster(p, neighbours)
        self.fitted = True

    def expand_cluster(self, p, neighbours):
        if self.n_clusters not in self.clusters:
            self.clusters[self.n_clusters] = []
        self.clustered.add(tuple(p))
        self.clusters[self.n_clusters].append(list(p))
        while neighbours:
            q = neighbours.pop()
            if tuple(q) not in self.visited:
                self.visited.add(tuple(q))
                q_neighbours = self.get_region(q)
                if len(q_neighbours) > self.min_samples:
                    neighbours.extend(q_neighbours)
            if tuple(q) not in self.clustered:
                self.clustered.add(tuple(q))
                self.clusters[self.n_clusters].append(q)
                if q in self.clusters[0]:
                    self.clusters[0].remove(q)
    def get_labels(self):
        labels = np.array([])
        if not self.fitted:
            self.fit()
        for data in self.dataset:
            for i in range(self.n_clusters + 1):
                if list(data) in self.clusters[i]:
                    labels = np.append(labels, i).astype(int)
        return labels

epsilon=[]
for k in range(1, 20):
    test = DB_SCAN(dataset, k*5, 2)
    epsilon.append(test.n_clusters)

plt.plot(range(1, 20), epsilon, marker='s')
plt.xlabel('$k$')
plt.ylabel('$J(C_k)$')

#test = DB_SCAN(dataset, 30, 2)
#pred = test.get_labels()

#plt.figure()
#plt.scatter(dataset[:,0], dataset[:,1], c=colors[pred])
plt.show()
