import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from time import time

digits = load_digits()
X_digits, y_digits = digits.data, digits.target

# split the dataset, apply grid-search
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits,\
                                       random_state=5, train_size=0.75)

svc = SVC()
C_values = np.logspace(-1, 1, num = 10, base = 10)
gamma_values = np.linspace(0.1, 1, 4)

estimator = GridSearchCV(svc, \
    {'C' : C_values, 'kernel' : ['rbf', 'linear', 'poly'],\
     'gamma' : gamma_values})
print('Applying Grid Search:')

t0 = time()
estimator.fit(X_train, y_train)
t1 = time()

print('Finished in %0.3fs' % (t1 - t0))

means = estimator.cv_results_['mean_test_score']
stds = estimator.cv_results_['std_test_score']
ranks = estimator.cv_results_['rank_test_score']
for rank, mean, std, params in zip(ranks, means, stds, estimator.cv_results_['params']):
  print("%d : %0.3f (+/-%0.03f) for %r" % (rank, mean, std * 2, params))

print(estimator.score(X_test, y_test))
