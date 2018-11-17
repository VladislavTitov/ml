import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

num = 10
dataset = np.array([[i, 2*i, np.random.uniform(-10, 10)] for i in range(num)])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c='red')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

dataset_cntr = dataset - dataset.mean()
covmat = np.cov(dataset, rowvar=False)

vals, vects = np.linalg.eig(covmat)
print(vals, vects, sep='\n\n')

# Так как первое собственное значение принимает нулевое значение при таких входных данных, где первые две координаты линейно зависимы, я беру два вектора, соответствующих остальным собственным значениям
vect1 = vects[1].reshape(3, -1)
vect2 = vects[2].reshape(3, -1)

coord1 = np.dot(dataset_cntr, vect1).reshape(1, -1)
coord2 = np.dot(dataset_cntr, vect2).reshape(1, -1)

plt.figure()
plt.scatter(coord1, coord2, c='green')
plt.show()




