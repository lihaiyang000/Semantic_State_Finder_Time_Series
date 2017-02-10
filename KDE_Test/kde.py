# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy.stats.distributions import norm
from IPython.display import HTML
def kde_scipy(x, x_grid, bandwidth, **kwargs):
	kde = gaussian_kde(x, bw_method=bandwidth/x.std(ddof=1), **kwargs) #**kwargs表示关键字参数 
	return kde.evaluate(x_grid)

def kde_sklearn(x, x_grid, bandwidth, **kwargs):
	kde = KernelDensity(bandwidth = bandwidth, **kwargs)
	# fit Fit the Kernel Density model on the data.
	kde.fit(x[:, np.newaxis]) #newaxis 可以新增维度, array([1., 2., 3.])  >>> a[:,np.newaxis] array([[ 1.], [ 2.], [ 3.]])
	pdf = kde.score_samples(x_grid[:, np.newaxis])
	return np.exp(pdf)
kde_functions = [kde_scipy, kde_sklearn]
kde_funcnames = ['scipy', 'scikit-learn']

#implements
x_grid = np.linspace(-4.5, 3.5, 1000)

# Draw points from a bimodal distribution in 1D
np.random.seed(0)
x = np.concatenate([norm(-1, 1.).rvs(400),norm(1, 0.3).rvs(100)])
pdf_true = (0.8 * norm(-1, 1).pdf(x_grid) + 0.2 * norm(1, 0.3).pdf(x_grid))

# Plot the three kernel density estimates
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(13, 3))
fig.subplots_adjust(wspace=0)

for i in range(2):
    pdf = kde_functions[i](x, x_grid, bandwidth=0.26)
    ax[i].plot(x_grid, pdf, color='red', alpha=0.5, lw=3)
    ax[i].fill(x_grid, pdf_true, ec='gray', fc='gray', alpha=0.4)
    ax[i].set_title(kde_funcnames[i])
    ax[i].set_xlim(-4.5, 3.5)
HTML("<font color='#666666'>Gray = True underlying distribution</font><br>"
     "<font color='6666ff'>Blue = KDE model distribution (500 pts)</font>")

plt.show()