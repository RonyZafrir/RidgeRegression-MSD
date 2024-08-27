'''
    Code for reproducing figure 2 - ridge regression cross-validation on the Million Song Dataset (msd)
'''

import matplotlib as mpl
import numpy as np
import pandas
from numpy import sqrt
import matplotlib.pyplot as plt
import time
mpl.use('tkAgg')
start_time = time.time()

# importing the data
msd = pandas.read_table('YearPredictionMSD.txt',delimiter=',', nrows=100000).values

# standardizing the data
def standardize(Y):
    return (Y - np.mean(Y)) / np.std(Y)


# choosing lambda for ridge regression using 5-fold CV:
for col in range(msd.shape[1]):
    # standardize each column (=feature)
    msd[:, col] = standardize(msd[:, col])

# matrix dimensions
m = msd.shape[0]
p = msd.shape[1] - 1
print("Num of data points is", m, "and Num of features is",p )

np.random.seed(130)
msd = np.random.permutation(msd) # shuffles the rows
n = 1000
K = 5
q = int(m / n) # num of segments that the whole data can be divided into
gamma = p / n
n_0 = n / K
steps = 40 # amount of distinct lambdas to evaluate
lbd_seq = np.linspace(0, 0.6, steps)
num_diff_subsets = 90
cv_error = np.zeros((steps, K))
n_1 = n - n_0
'''
for the error bar, we take n = 1000, p = 90, K = 5, and average over 90 different sub-datasets
'''
for t in range(num_diff_subsets):
    X = msd[n * t: n * (t + 1), 1:]
    Y = msd[n * t: n * (t + 1), 0].reshape(n, 1)
    all_indices = set(np.arange(0, n, 1, dtype=int))
    # performing 5-fold CV for each of the 40 optional lambdas
    for i in range(steps):
        curr_lbd = lbd_seq[i]
        for j in range(K):
            test_idx = np.arange(j * n_0, (j + 1) * n_0, 1, dtype=int)
            X_test = X[test_idx, :] # 200x90
            Y_test = Y[test_idx, :]
            train_idx = list(all_indices - set(test_idx))
            X_train = X[train_idx, :] # 800x90
            Y_train = Y[train_idx, :]
            # computing beta hat -k
            beta_hat = np.linalg.inv(X_train.T @ X_train + curr_lbd  * n_1 * np.identity(p)) @ X_train.T @ Y_train
            # cv error for isotropic covariance (sigma_mat = I)
            # divided by num_diff_subsets since we do it for each
            cv_error[i, j] += (np.linalg.norm(Y_test - X_test @ beta_hat) ** 2 / n_0 ) / num_diff_subsets


lbd_cv_idx = np.argmin(np.mean(cv_error, 1))
lbd_cv = lbd_seq[lbd_cv_idx]
lbd_cv_debiased = lbd_cv * (K - 1) / K # the bias-corrected parameter that's proposed in the paper
lbd_cv_debiased_idx = 0
for i in range(steps):
    if lbd_seq[i] >= lbd_cv_debiased:
        lbd_cv_debiased_idx = i
        break


'''
for the test error, we train on 1000 data points and fit on 9000 test datapoints

'''
X_train = msd[n * (q - 10): n * (q - 9), 1:] #1000x90
Y_train = msd[n * (q - 10): n * (q - 9), 0]
X_test = msd[n * (q - 9): n * q, 1:] #9000x90
Y_test = msd[n * (q - 9): n * q, 0]
test_error = np.zeros(steps)

for i in range(steps):
    lbd = lbd_seq[i]
    beta_ridge = np.linalg.inv(X_train.T @ X_train + lbd * np.identity(p) * n ) @ X_train.T @ Y_train
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
print("My program took", time.time() - start_time, " seconds to run")
plt.show()
print(test_error[lbd_cv_debiased_idx], test_error[lbd_cv_idx], test_error[lbd_smallest_idx])
print("Debias improve test error by", test_error[lbd_cv_idx] - test_error[lbd_cv_debiased_idx])

