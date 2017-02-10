# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
#from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy.stats.distributions import norm

def kde_sklearn(x, x_grid, bandwidth, **kwargs):
	kde = KernelDensity(bandwidth = bandwidth, **kwargs)
	# fit Fit the Kernel Density model on the data.
	kde.fit(x[:, np.newaxis]) #newaxis 可以新增维度, array([1., 2., 3.])  >>> a[:,np.newaxis] array([[ 1.], [ 2.], [ 3.]])
	pdf = kde.score_samples(x_grid[:, np.newaxis])
	return np.exp(pdf)

x_grid = np.linspace(6000, 7000, 10000)

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