import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets

from sklearn import decomposition

iris = datasets.load_iris()

X = iris.data
y = iris.target

fig = plt.figure(1, figsize=(6, 5))

ax = Axes3D(fig, elev=48, azim=134)

for name, label in [('Setosa',0), ('Versicolour',1), ('Verginics', 2)]:
    ax.text3D(X[y==label,0].mean(), X[y==label,1].mean()+1, X[y==label,2].mean(), name)

y_clr = np.choose(y, [1,2,0]).astype(np.float)
ax.scatter(X[:,0], X[:,1], X[:,2], c=y_clr, cmap=plt.cm.nipy_spectral)
plt.show()


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict_proba(X_test)
score = accuracy_score(y_test, pred.argmax(axis=1))
print(score)

pca = decomposition.PCA(n_components=2)
X_centered = X - X.mean(axis=0)
pca.fit(X_centered)
X_pca = pca.transform(X_centered)

plt.plot(X_pca[y==0, 0], X_pca[y==0,1], 'bo', label='Setosa')
plt.plot(X_pca[y==1, 0], X_pca[y==1,1], 'go', label='Versicolor')
plt.plot(X_pca[y==2, 0], X_pca[y==2,1], 'ro', label='Verginica')
plt.legend(loc=0)
plt.show()



X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=.3, random_state=42)
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict_proba(X_test)
score = accuracy_score(y_test, pred.argmax(axis=1))
print(score)


num = 13
dataset = np.array([[i, 2*i+np.random.uniform(-3, 3)] for i in range(num)])
print(dataset)

plt.figure()
plt.scatter(dataset[:, 0], dataset[:, 1], c='black')
plt.show()
