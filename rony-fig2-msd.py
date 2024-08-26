"""
    Code for reproducing figure 2 - ridge regression cross-validation on the Million Song Dataset (msd)
"""

import matplotlib as mpl
import numpy as np
import pandas as pd
from numpy import sqrt
import matplotlib.pyplot as plt
mpl.use('tkAgg')

# importing the data
msd = pd.read_table('C:/Users/ronyz/Documents/לימודים/נושאים נבחרים סטטיסטיקה חישובית/ridgeRegression/YearPredictionMSD.txt',
                    delimiter=',', nrows=100000).values

# standardizing the data
def standardize(Y):
    return (Y - np.mean(Y)) / np.std(Y)

# choosing lambda for ridge regression using 5-fold CV:
m = msd.shape[0]
p = msd.shape[1] - 1
for i in range(msd.shape[1]):
    # standardize each column i.e. each feature
    msd[:, i] = standardize(msd[:, i])
np.random.seed(130)
msd = np.random.permutation(msd)
n = 1000
K = 5
q = int(m / n) # num of segments that the data can be divided into
gamma = p / n
batch_size = n / K
steps = 40 # amount of distinct lambdas to evaluate
lbd_seq = np.linspace(0, 0.6, steps)
cv_error = np.zeros((steps, K))



'''
cross validation

for the error bar, we average over 90 = q - 10 different sub-datasets
for the test error, we train on 1000 data points and fit on 9000 test datapoints
'''
for k in range(q - 10):
    X = msd[n * k: n * (k + 1), 1:]
    Y = msd[n * k: n * (k + 1), 0].reshape(n, 1)
    for i in range(steps):
        lbd = lbd_seq[i]
        for j in range(K):
            test_idx = np.arange(j * batch_size, (j + 1) * batch_size, 1, dtype=int)
            X_test = X[test_idx, :] # 200x90
            Y_test = Y[test_idx, :]
            train_idx = list(set(np.arange(0, n, 1, dtype=int)) - set(test_idx))
            X_train = X[train_idx, :] # 800x90
            Y_train = Y[train_idx, :]
            beta_hat = np.linalg.inv(X_train.T @ X_train / (n - batch_size) + lbd * np.identity(p)) @ X_train.T @ Y_train / (n - batch_size)
            cv_error[i, j] = cv_error[i, j] + np.linalg.norm(Y_test - X_test @ beta_hat) ** 2 / batch_size / (q - 10)


lbd_cv_idx = np.argmin(np.mean(cv_error, 1))
lbd_cv = lbd_seq[lbd_cv_idx]
lbd_cv_debiased = lbd_cv * (K - 1) / K # the bias-corrected parameter that's proposed in the paper
lbd_cv_debiased_idx = 0
for i in range(steps):
    if lbd_seq[i] >= lbd_cv_debiased:
        lbd_cv_debiased_idx = i
        break

# test error
X_train = msd[n * (q - 10): n * (q - 9), 1:] #1000x90
Y_train = msd[n * (q - 10): n * (q - 9), 0]
X_test = msd[n * (q - 9): n * q, 1:] #9000x90
Y_test = msd[n * (q - 9): n * q, 0]
test_error = np.zeros(steps)

for i in range(steps):
    lbd = lbd_seq[i]
    beta_ridge = np.linalg.inv(
        X_train.T @ X_train / n + lbd * np.identity(p)) @ X_train.T @ Y_train / n
    test_error[i] = np.linalg.norm(Y_test - X_test @ beta_ridge) ** 2 / (9 * n)

lbd_smallest_idx = np.argmin(test_error)
lbd_smallest = lbd_seq[lbd_smallest_idx]


lbd_theory = gamma
lbd_theory_idx = 0
for i in range(steps):
    if lbd_seq[i] >= lbd_theory:
        lbd_theory_idx = i
        break

# Figure 2 (left), cross-validation on the million song dataset
lb = 0.83
ub = 0.91
plt.errorbar(lbd_seq, np.mean(cv_error, 1), np.sqrt(np.var(cv_error, 1)), capsize=2, label='CV error bar')
plt.plot(lbd_seq, test_error, label='Test error')
plt.plot(lbd_cv * np.ones(10), np.linspace(lb, ub, 10), ls='--', linewidth=3, label='CV min {:.3f}'.format(test_error[lbd_cv_idx]))
plt.plot(lbd_cv_debiased * np.ones(10), np.linspace(lb, ub, 10), ls='-.', linewidth=3, label='Debiased CV {:.3f}'.format(test_error[lbd_cv_debiased_idx]))
plt.plot(lbd_smallest * np.ones(10), np.linspace(lb, ub, 10), ls=':', label='Test error min {:.3f}'.format(test_error[lbd_smallest_idx]), linewidth=3)
plt.plot(lbd_theory * np.ones(10), np.linspace(lb, ub, 10), ls=':', label='Theory {:.3f}'.format(test_error[lbd_theory_idx]), linewidth=3)
plt.legend(fontsize=13)
plt.grid(linestyle='dotted')
plt.xlabel(r'$\lambda$', fontsize=13)
plt.ylabel('CV test error', fontsize=13)
plt.title('MSD CV', fontsize=13)
plt.show()



