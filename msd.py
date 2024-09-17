# Code for reproducing figure 2 - Ridge Regression Cross-Validation on the Million Song Dataset (msd)


import matplotlib as mpl
import numpy as np
import pandas
import matplotlib.pyplot as plt
import time
mpl.use('tkAgg')
start_time = time.time()

def standardize(Y):
    return (Y - np.mean(Y)) / np.std(Y)

# importing the data
msd = pandas.read_table('YearPredictionMSD.txt',delimiter=',', nrows=100000).values

for col in range(msd.shape[1]):
    # standardize each column (=feature)
    msd[:, col] = standardize(msd[:, col])

m = msd.shape[0]
p = msd.shape[1] - 1
#print("Num of data points is", m, "and Num of features is",p )
np.random.seed(130)
msd = np.random.permutation(msd) # shuffles the rows
# variables we'll need, matches values from the paper:
n = 1000
K = 5
n_0 = n / K # batch size
steps = 40 # amount of distinct lambdas to evaluate
lbd_seq = np.linspace(0, 0.6, steps)
num_diff_subsets = 90
cv_error = np.zeros((steps, K, num_diff_subsets))
n_1 = n - n_0


'''
To "retrace their steps"- we'll divide the data like so:
90000 data points for the error bars
10000 data points for the test error
 
let's start with reproducing results needed for the error bar:
we'll find the CV min and the debiased CV and its (approximated) error 

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
            cv_error[i, j, t] = np.linalg.norm(Y_test - X_test @ beta_hat) ** 2 #/ n_0
    #to avoid numerical error; for each possible lambda we'll sum cv_error from smallest to largest.
    #Note: I'll divided by n_0*num_diff_subsets in the end to be as frugal as possible with
    cv_sorted_err = np.zeros((steps,K))
    denominator = n_0*num_diff_subsets
    for i in range(steps):
        for j in range(K):
            s_to_l = np.sort(cv_error[i,j,:]) # sorts cv_error for each subset, from the smallest error to largest
            cv_sorted_err[i,j] = np.sum(s_to_l) / (n_0*num_diff_subsets)


lbd_cv_idx = np.argmin(np.mean(cv_sorted_err, 1)) # axis=1 -> mean over the 5-folds
lbd_cv = lbd_seq[lbd_cv_idx] # the lambda that got us the smallest cv error
lbd_cv_debiased = lbd_cv * (K - 1) / K # the bias-corrected parameter that's proposed in the paper
'''
now we should find the cv error for that bias-corrected lambda:
we'll do it by finding the cv error for the smallest lambda that's bigger than it out of the ones we've tested 
'''
lbd_cv_debiased_idx = 0
for i in range(steps):
    if lbd_seq[i] >= lbd_cv_debiased:
        lbd_cv_debiased_idx = i
        break


#for the test error, we train on 1000 data points and fit on 9000 test datapoints
q = int(m/n)
X_train = msd[n * (q - 10): n * (q - 9), 1:] #1000x90
Y_train = msd[n * (q - 10): n * (q - 9), 0]
X_test = msd[n * (q - 9): n * q, 1:] #9000x90
Y_test = msd[n * (q - 9): n * q, 0]
test_error = np.zeros(steps)

for i in range(steps): # for each lambda
    beta_ridge = np.linalg.inv(X_train.T @ X_train + lbd_seq[i] * n * np.identity(p)) @ X_train.T @ Y_train
    test_error[i] = np.linalg.norm(Y_test - X_test @ beta_ridge) ** 2 / (9 * n)

lbd_smallest_idx = np.argmin(test_error)
lbd_smallest = lbd_seq[lbd_smallest_idx]

lbd_theory = p / n # as was noted in the paper
lbd_theory_idx = 0
# exactly like we did for the debiased lambda:
for i in range(steps):
    if lbd_seq[i] >= lbd_theory:
        lbd_theory_idx = i
        break


'''
figure 2
'''
lb = 0.83
ub = 0.92
plt.errorbar(lbd_seq, np.mean(cv_sorted_err, 1), np.sqrt(np.var(cv_sorted_err, 1)), capsize=2, label='CV errorbar')
plt.plot(lbd_seq, test_error, label='Test error')
plt.plot(lbd_cv * np.ones(10), np.linspace(lb, ub, 10), ls='--', linewidth=3, label='CV min {:.3f}'.format(test_error[lbd_cv_idx]))
plt.plot(lbd_cv_debiased * np.ones(10), np.linspace(lb, ub, 10), ls='-.', linewidth=3, label='Debiased CV {:.3f}'.format(test_error[lbd_cv_debiased_idx]))
plt.plot(lbd_smallest * np.ones(10), np.linspace(lb, ub, 10), ls=':', label='Test error min {:.3f}'.format(test_error[lbd_smallest_idx]), linewidth=3)
plt.plot(lbd_theory * np.ones(10), np.linspace(lb, ub, 10), ls=':', label='Theory {:.3f}'.format(test_error[lbd_theory_idx]), linewidth=3)
plt.legend(fontsize = 12)
plt.grid(linestyle = 'dotted')
plt.xlabel(r'$\lambda$', fontsize = 13)
plt.ylabel('CV test error', fontsize = 13)
plt.title('MSD CV', fontsize=15)
#print("My program took", time.time() - start_time, " seconds to run")
plt.savefig("./CV_msd.png")
plt.show()
