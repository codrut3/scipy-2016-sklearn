from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

iris = load_iris()

print(iris.keys())

n_samples, n_features = iris.data.shape
print('Number of samples:', n_samples)
print('Number of features:', n_features)
print(iris.data[0])

print(iris.data.shape)
print(iris.target.shape)

print(iris.target)

print(np.bincount(iris.target))

print(iris.target_names)

x_index = 0

colors = ['blue', 'red', 'green']
'''
for label, color in zip(range(len(iris.target_names)), colors):
  plt.hist(iris.data[iris.target == label, x_index],\
    label = iris.target_names[label], color = color)

plt.xlabel(iris.feature_names[x_index])

plt.legend(loc = 'upper right')
plt.show()
'''

x_index = 3
y_index = 1

'''
for label, color in zip(range(len(iris.target_names)), colors):
  plt.scatter(iris.data[iris.target == label, x_index], \
    iris.data[iris.target == label, y_index], \
    label = iris.target_names[label],\
    c = color)

plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.legend(loc = 'upper left')
plt.show()
'''

iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
pd.tools.plotting.scatter_matrix(iris_df, figsize=(8, 8))
plt.show()
