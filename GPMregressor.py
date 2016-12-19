"""
===========================
Gaussian Process Regression Model Generator

This code is generator from several
===========================
A. Rysanek   script development       12.12.2016


"""

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

__author__ = "Adam Rysanek"
__copyright__ = "Copyright 2016, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Adam Rysanek"]
__license__ = "MIT"
__version__ = "0.1"


# Load model data
# --------------------------------------------------------------------------
xl = pd.ExcelFile("Sinwave_Test.xlsx")
print xl.sheet_names
df = xl.parse("Sheet2")
print df.head()

X = df.iloc[0:2000,1:4].values # Input values per sample
y = df.iloc[0:2000,7].values # Model output per sample (e.g., RMSE)

min_max_scaler = preprocessing.MinMaxScaler() # Normalize input data
Xnorm = min_max_scaler.fit_transform(X)
print X[29,:]
print y[29]

x = np.atleast_2d([[10.,6.,3.],[6,8,4],[4,4,2],[5,3,2],[6,4,9]]) # Only used for testing at end of model.
xpred = Xnorm[0:5,:] # only used for testing at end of model

# Instantiate  Gaussian Process model
# --------------------------------------------------------------------------------------
# The default kernal hyperparameters here below come from a previous analyis of another model. There may be a positive / negative
# effect on GP model convergence based on what starting hyperparameters your provide. This requires further investigation,
# though I haven't noticed a significant problem
# The structure of the GP model I've used is taken directly from the "Moana Loa GPR" example found in the scikit documentation

k1 = 316**2 * RBF(length_scale=1e-5)  # long term smooth rising trend
k2 = 0.0156**2 * RBF(length_scale=15.4) * ExpSineSquared(length_scale=2.38e+3, periodicity=0.0291)  # seasonal component
# medium term irregularity
k3 = 316**2 * RationalQuadratic(length_scale=1.86, alpha=0.199)
k4 = 316**2 * RBF(length_scale=3.57e-05) + WhiteKernel(noise_level=1e+5)  # noise terms
kernel = k1 + k2 + k3 + k4

# Generate GP Model
# -------------------------------------------------------------------------------------
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-7,
                              normalize_y=True, n_restarts_optimizer=2) # Play with values for n_restarts_optimizer. I've gone as high as 9 when starting a new regression)
gp.fit(Xnorm, y)

print("\nLearned kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))

# Save Gaussian Process Model to Pickle
# --------------------------------------------------------------------------------------
joblib.dump(gp, 'SinusGP.pkl')


# Post processing / Not necessary but just for testing
# --------------------------------------------------------------------------------------

X_ = np.linspace(X.min(), X.max() + 30, 1000)[:, np.newaxis]
y_pred, y_std = gp.predict(Xnorm, return_std=True)

line = plt.figure()
plt.plot(y, y_pred, "o")
plt.plot([0, 0], [100, 100], 'k-', lw=2)
plt.show()

y_pred, y_std = gp.predict(Xnorm[0,:], return_std=True)
print y_pred
print y_std
