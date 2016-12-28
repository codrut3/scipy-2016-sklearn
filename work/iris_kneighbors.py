from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()

print(iris.target)

X_train, X_test, y_train, y_test = train_test_split(iris.data, \
                                      iris.target, train_size = 0.5, \
                                      random_state = 123, \
                                      stratify = iris.target)

knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))

y_pred = knn.predict(X_test)
print(np.mean(y_pred == y_test))

print('Samples correctly classified:')
correct_idx = np.where(y_pred == y_test)[0]
print(correct_idx)

print('\nSamples incorrectly classified:')
incorrect_idx = np.where(y_pred != y_test)[0]
print(incorrect_idx)

colors = ["darkblue", "darkgreen", "gray"]

for n, color in enumerate(colors):
  idx = np.where(y_test == n)[0]
  plt.scatter(X_test[idx, 1], X_test[idx, 2], color = color, label = 'Class %s' % str(n))

wrong_colors = ["darkred"]

for i in range(len(incorrect_idx)):
  plt.scatter(X_test[incorrect_idx[i], 1],\
    X_test[incorrect_idx[i], 2], color = wrong_colors[0])

plt.xlabel('sepal width [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc=3)
plt.title("Iris Classification results")
plt.show()
