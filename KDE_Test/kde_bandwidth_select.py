from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import numpy as np
from scipy.stats.distributions import norm
x = np.concatenate([norm(-1, 1.).rvs(400),norm(1, 0.3).rvs(100)])
y = np.concatenate([norm(0, 1.).rvs(400),norm(1, 0.3).rvs(100)])
grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(-4.5, 3.5, 100)},cv=20)#np.linspace  Return evenly spaced numbers over a specified interval. # 20-fold cross-validation
grid.fit(x[:, None])
print grid.best_params_