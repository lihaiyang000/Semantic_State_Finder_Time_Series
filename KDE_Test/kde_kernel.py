# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy.stats.distributions import norm

def kde_sklearn(x, x_grid, bandwidth, **kwargs):
	kde = KernelDensity(bandwidth = bandwidth, **kwargs)
	# fit Fit the Kernel Density model on the data.
	kde.fit(x[:, np.newaxis]) #newaxis 可以新增维度, array([1., 2., 3.])  >>> a[:,np.newaxis] array([[ 1.], [ 2.], [ 3.]])
	pdf = kde.score_samples(x_grid[:, np.newaxis])
	return np.exp(pdf)

def kde_sklearn_tophat(x, x_grid, bandwidth, **kwargs): #矩形核函数
    kde = KernelDensity(bandwidth = bandwidth, kernel='tophat', **kwargs)
    # fit Fit the Kernel Density model on the data.
    kde.fit(x[:, np.newaxis]) #newaxis 可以新增维度, array([1., 2., 3.])  >>> a[:,np.newaxis] array([[ 1.], [ 2.], [ 3.]])
    pdf = kde.score_samples(x_grid[:, np.newaxis])
    return np.exp(pdf)

def kde_sklearn_epanechnikov(x, x_grid, bandwidth, **kwargs): #矩形核函数
    kde = KernelDensity(bandwidth = bandwidth, kernel='epanechnikov', **kwargs)
    # fit Fit the Kernel Density model on the data.
    kde.fit(x[:, np.newaxis]) #newaxis 可以新增维度, array([1., 2., 3.])  >>> a[:,np.newaxis] array([[ 1.], [ 2.], [ 3.]])
    pdf = kde.score_samples(x_grid[:, np.newaxis])
    return np.exp(pdf)

def kde_sklearn_linear(x, x_grid, bandwidth, **kwargs): #矩形核函数
    kde = KernelDensity(bandwidth = bandwidth, kernel='linear', **kwargs)
    # fit Fit the Kernel Density model on the data.
    kde.fit(x[:, np.newaxis]) #newaxis 可以新增维度, array([1., 2., 3.])  >>> a[:,np.newaxis] array([[ 1.], [ 2.], [ 3.]])
    pdf = kde.score_samples(x_grid[:, np.newaxis])
    return np.exp(pdf)

def kde_sklearn_cosine(x, x_grid, bandwidth, **kwargs): #矩形核函数
    kde = KernelDensity(bandwidth = bandwidth, kernel='cosine', **kwargs)
    # fit Fit the Kernel Density model on the data.
    kde.fit(x[:, np.newaxis]) #newaxis 可以新增维度, array([1., 2., 3.])  >>> a[:,np.newaxis] array([[ 1.], [ 2.], [ 3.]])
    pdf = kde.score_samples(x_grid[:, np.newaxis])
    return np.exp(pdf)

def kde_sklearn_exp(x, x_grid, bandwidth, **kwargs): #矩形核函数
    kde = KernelDensity(bandwidth = bandwidth, kernel='exponential', **kwargs) #e 指数
    # fit Fit the Kernel Density model on the data.
    kde.fit(x[:, np.newaxis]) #newaxis 可以新增维度, array([1., 2., 3.])  >>> a[:,np.newaxis] array([[ 1.], [ 2.], [ 3.]])
    pdf = kde.score_samples(x_grid[:, np.newaxis])
    return np.exp(pdf)

kde_functions = [kde_sklearn, kde_sklearn_tophat, kde_sklearn_epanechnikov, kde_sklearn_linear, kde_sklearn_cosine, kde_sklearn_exp]
kde_funcnames = ['gaussian', 'tophat', 'epanechnikov', 'linear', 'cosine', 'exponential']

#implements
x_grid = np.linspace(-4.5, 3.5, 100)

# Draw points from a bimodal distribution in 1D
np.random.seed(0)
x = np.concatenate([norm(-1, 1.).rvs(400),norm(1, 0.3).rvs(100)])
pdf_true = (0.8 * norm(-1, 1).pdf(x_grid) + 0.2 * norm(1, 0.3).pdf(x_grid))

# Plot the three kernel density estimates
fig, ax = plt.subplots(1, 6, sharey=True, figsize=(13, 6))
fig.subplots_adjust(wspace=0)

for i in range(6):
    pdf = kde_functions[i](x, x_grid, bandwidth=0.26)
    ax[i].plot(x_grid, pdf, color='green', alpha=0.5, lw=1)
    ax[i].hist(x, 40, fc='black', histtype='stepfilled', alpha=0.2, normed=True)
    ax[i].set_title(kde_funcnames[i])
    ax[i].set_xlim(-4.5, 3.5)
plt.show()