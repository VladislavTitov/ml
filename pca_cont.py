import numpy as np
import matplotlib.pyplot as plt

num = 13
dataset = np.array([[i, 2*i+np.random.uniform(-3, 3)] for i in range(num)])

#plt.figure()
#plt.scatter(dataset[:, 0], dataset[:, 1], c='black')
#plt.show()

dataset_cntr = dataset - dataset.mean(axis=0)

covmat = np.cov(dataset_cntr, rowvar=False)
vals, vects = np.linalg.eig(covmat)

vect1 = vects[0].reshape(2, -1)
vect2 = vects[1].reshape(2, -1)

coord1 = np.dot(dataset_cntr, vect1)
coord2 = np.dot(dataset_cntr, vect2)

plt.figure()
plt.scatter(coord1[:, 0], [0 for i in range(num)], c='red')
plt.show()

plt.figure()
plt.scatter(coord2[:, 0], [0 for i in range(num)], c='green')
plt.show()

dot_prod = np.array([i[0]*vect2[0] + i[1]*vect2[1] for i, j in zip(dataset_cntr, vect1)])
print(dot_prod)

