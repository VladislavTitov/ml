from matplotlib.image import imread, imsave
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np

def get_image(img_path, show=True):
    orig_img = imread(img_path)
    
    if show:
        plt.imshow(orig_img)
        plt.show()
        print('Shape:', orig_img.shape)
    return orig_img

def get_kmeans(orig_img, n_colors=8):
    X = orig_img.reshape((-1, 4))
    print('Shape after reshape:', orig_img.shape)
    kmeans = KMeans(n_clusters=n_colors).fit(X)
    pred = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    return centroids[pred].reshape(orig_img.shape)

n_colors = 4
all_img = []
orig_img = get_image('test.png', False)
new_img = get_kmeans(orig_img, n_colors)
all_img += [orig_img, new_img]

fig, axarr = plt.subplots(nrows=1, ncols=2, sharex=True)
axarr[0] = plt.imshow(all_img[0])
axarr[1] = plt.imshow(all_img[1])
fig.tight_layout()
plt.show()

