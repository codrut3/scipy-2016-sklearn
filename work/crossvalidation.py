from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit

iris = load_iris()
X, y = iris.data, iris.target

classifier = KNeighborsClassifier()

rng = np.random.RandomState(0)

permutation = rng.permutation(len(X))
X, y = X[permutation], y[permutation]
print(y)

k = 5
n_samples = len(X)
fold_size = n_samples // k
scores = []
masks = []
for fold in range(k):
    # generate a boolean mask for the test set in this fold
    test_mask = np.zeros(n_samples, dtype=bool)
    test_mask[fold * fold_size : (fold + 1) * fold_size] = True
    # store the mask for visualization
    masks.append(test_mask)
    # create training and test sets using this mask
    X_test, y_test = X[test_mask], y[test_mask]
    X_train, y_train = X[~test_mask], y[~test_mask]
    # fit the classifier
    classifier.fit(X_train, y_train)
    # compute the score and record it
    scores.append(classifier.score(X_test, y_test))

#plt.matshow(masks, cmap='gray_r')
#plt.show()
print(scores)
print(np.mean(scores))

scores = cross_val_score(classifier, X, y, cv = 5)
print(scores)
print(np.mean(scores))

def plot_cv(cv, features, labels):
    masks = []
    for train, test in cv.split(features, labels):
        mask = np.zeros(len(labels), dtype=bool)
        mask[test] = 1
        masks.append(mask)
    
    plt.matshow(masks, cmap='gray_r')
    plt.show()

#plot_cv(ShuffleSplit(n_splits=5, test_size=.2), iris.data, iris.target)
cv = ShuffleSplit(n_splits=5, test_size=.2)
print('ShuffleSplit: ')
print(cross_val_score(classifier, X, y, cv=cv))
