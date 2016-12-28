import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     random_projection)

def digits_plot(digits, labels):
    n_digits = 500
    X = digits.data[:n_digits]
    y = digits.target[:n_digits]
    n_samples, n_features = X.shape
    n_neighbors = 30

    def plot_embedding(X, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        plt.figure()
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                     color=plt.cm.Set1(labels[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})

        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 1e5:
                    # don't show points that are too close
                    # set a high threshold to basically turn this off
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                    X[i])
                ax.add_artist(imagebox)
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)

    print("Computing PCA projection")
    pca = decomposition.PCA(n_components=2).fit(X)
    X_pca = pca.transform(X)
    plot_embedding(X_pca, "Principal Components projection of the digits")
    plt.show()


digits = load_digits()
X = digits.data
y = digits.target

kmeans = KMeans(n_clusters = 10, random_state = 42)
labels = kmeans.fit_predict(X)
print(adjusted_rand_score(y, labels))

fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1,\
  hspace=0.05, wspace=0.05)

# plot the cluster centers: each image is 8x8 pixels
n_centers = 10
for i in range(n_centers):
  ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
  ax.imshow(kmeans.cluster_centers_[i].reshape(8, 8), cmap=plt.cm.binary, \
    interpolation='bicubic')

digits_plot(digits, labels)
