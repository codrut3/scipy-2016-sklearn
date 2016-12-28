from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

boston = load_boston()
X = boston.data
y = boston.target

print('Number of features is %d' % X.shape[1])
print('Number of samples is %d' % X.shape[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75,\
                                  random_state = 42)

print('Percent size of train set is %f' % (float(len(X_train)) / len(X)))

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_lin = lin_reg.predict(X_test)

print('The R^2 score of linear regression is %f' % (lin_reg.score(X_test, y_test)))

neigh_reg = KNeighborsRegressor(n_neighbors = 3)
neigh_reg.fit(X_train, y_train)

print('The R^2 score of k-neighbors regression is %f' % (neigh_reg.score(X_test, y_test)))
