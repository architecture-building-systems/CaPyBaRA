from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared

import pickle
from sklearn.externals import joblib

from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
import time

xl = pd.ExcelFile("OutputSummary_17-01-15-014432.xlsx")
print xl.sheet_names
df = xl.parse("OutputSummary_17-01-15-014432")
print df.head()
#df = pd.read_csv("OutputSummary_16-12-05-004805.csv")

X = df.iloc[0:4999,1:36].values
#Xnorm = preprocessing.normalize(X)
min_max_scaler = preprocessing.MinMaxScaler()
Xnorm = min_max_scaler.fit_transform(X)
y = df.iloc[0:4999,47].values
print X[29,:]
print y[29]

x = np.atleast_2d([[10.,6.,3.],[6,8,4],[4,4,2],[5,3,2],[6,4,9]])
xpred = Xnorm[0:5,:]

# Instantiate a Gaussian Process model
#kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e6))
#gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
#ker_rbf = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")

# Kernel with parameters given in GPML book
k1 = 5**2 * RBF(length_scale=1e-5)  # long term smooth rising trend
k2 = 5**2 * RBF(length_scale=0.000415) * ExpSineSquared(length_scale=3.51e-5, periodicity=0.000199)  # seasonal component
# medium term irregularity
k3 = 316**2 * RationalQuadratic(length_scale=3.54, alpha=1e+05)
k4 = 316**2 * RBF(length_scale=4.82) + WhiteKernel(noise_level=0.43)  # noise terms
kernel = k1 + k2 + k3 + k4

gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-7,
                              normalize_y=True, n_restarts_optimizer=0)
gp.fit(Xnorm, y)

print("\nLearned kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))

X_ = np.linspace(X.min(), X.max() + 30, 1000)[:, np.newaxis]
y_pred, y_std = gp.predict(Xnorm, return_std=True)

joblib.dump(gp, 'TRNSYSgp.pkl')
#pickle.dump(gp, open('TRNSYSgppkl.pkl','wb'))

line = plt.figure()
plt.plot(y, y_pred, "o")
plt.plot([0, 0], [100, 100], 'k-', lw=2)
plt.show()

y_pred, y_std = gp.predict(Xnorm[0,:], return_std=True)
print y_pred
print y_std

y_pred, y_std = gp.predict(Xnorm[1,:], return_std=True)
print y_pred
print y_std

y_pred, y_std = gp.predict(Xnorm[2,:], return_std=True)
print y_pred
print y_std

y_pred, y_std = gp.predict(Xnorm[3,:], return_std=True)
print y_pred
print y_std

y_pred, y_std = gp.predict(Xnorm[4,:], return_std=True)
print y_pred
print y_std

