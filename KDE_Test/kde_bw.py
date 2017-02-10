# coding: utf-8
from sklearn.neighbors import KernelDensity
from scipy.stats.distributions import norm
import matplotlib.pyplot as plt
import numpy as np
def kde_sklearn(x, x_grid, bandwidth, **kwargs):
	kde = KernelDensity(bandwidth = bandwidth, **kwargs)
	# fit Fit the Kernel Density model on the data.
	kde.fit(x[:, np.newaxis]) #newaxis 可以新增维度, array([1., 2., 3.])  >>> a[:,np.newaxis] array([[ 1.], [ 2.], [ 3.]])
	pdf = kde.score_samples(x_grid[:, np.newaxis])
	return np.exp(pdf)

x_grid = np.linspace(-4.5, 3.5, 1000)

# Draw points from a bimodal distribution in 1D
np.random.seed(0)
x = np.concatenate([norm(-1, 1.).rvs(400),norm(1, 0.3).rvs(100)])
pdf_true = (0.8 * norm(-1, 1).pdf(x_grid) + 0.2 * norm(1, 0.3).pdf(x_grid))

fig, ax = plt.subplots()
for bandwidth in [0.1, 0.3, 0.4, 1.0]:
	ax.plot(x_grid, kde_sklearn(x, x_grid, bandwidth=bandwidth),label='bw={0}'.format(bandwidth), linewidth=4, alpha=0.5)
ax.hist(x, 40, fc='black', histtype='stepfilled', alpha=0.2, normed=True)
ax.set_xlim(-4.5,3.5)
ax.legend(loc="upper right")
plt.show()
