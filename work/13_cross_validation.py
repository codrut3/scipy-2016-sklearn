from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

iris = load_iris()
X, y = iris.data, iris.target

classifier = KNeighborsClassifier()

cv = KFold(n_splits = 3)

print(cross_val_score(classifier, X, y, cv=cv))

def plot_cv(cv, features, labels):
    masks = []
    for train, test in cv.split(features, labels):
        mask = np.zeros(len(labels), dtype=bool)
        mask[test] = 1
        masks.append(mask)
    
    plt.matshow(masks, cmap='gray_r')
    plt.show()

plot_cv(cv, X, y)
