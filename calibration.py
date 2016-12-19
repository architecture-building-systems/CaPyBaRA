"""
===========================
Bayesian calibration routine

based on work of (Patil_et_al, 2010 - #http://www.map.ox.ac.uk/media/PDF/Patil_et_al_2010.pdf) in MCMC in pyMC3
and the works of bayesian calibration of (Kennedy and O'Hagan, 2001)
===========================
A. Rysanek / J. Fonseca  script development       12.12.2016


"""

from __future__ import division

import pandas as pd
import pymc3 as pm
from pymc3.backends import SQLite
import theano.tensor as tt
import theano
# from cea.demand import demand_main
import matplotlib.pyplot as plt
import random
from sklearn.externals import joblib
import os
from sklearn import preprocessing
import numpy as np
from scipy import optimize
from theano import as_op

# import cea.globalvar as gv
# gv = gv.GlobalVariables()
# import cea.inputlocator as inputlocator
# scenario_path = gv.scenario_reference
# locator = inputlocator.InputLocator(scenario_path=scenario_path)
# weather_path = locator.get_default_weather()

__author__ = "Adam Rysanek, Jimeno A. Fonseca"
__copyright__ = "Copyright 2016, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Adam Rysanek, Jimeno A. Fonseca"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Daren Thomas"
__email__ = "cea@arch.ethz.ch"
__status__ = "Production"


# Load Gaussian Process Model
# ------------------------------------------------------------------------------------------------------
gp = joblib.load('SinusGP.pkl')

# Reload Input Data and re-normalize for calibration. This entire block of code is not ultimately necessary.
# Normalization could be done directly in the input dataset (e.g., Excel). However, this code exsts mainly because
# I was troubleshooting different approaches originally
# ------------------------------------------------------------------------------------------------------
xl = pd.ExcelFile("Sinwave_Test.xlsx")
print xl.sheet_names
df = xl.parse("Sampled_Inputs_withRMSE")
print df.head()
#df = pd.read_csv("OutputSummary_16-12-05-004805.csv")

X = df.iloc[0:2001,1:4].values
Xnorm = preprocessing.normalize(X)
min_max_scaler = preprocessing.MinMaxScaler()
Xnorm = min_max_scaler.fit_transform(X)
y = df.iloc[0:2001,7].values

min1 = min(Xnorm[:,0])
max1 = max(Xnorm[:,0])
min2 = min(Xnorm[:,1])
max2 = max(Xnorm[:,1])
min3 = min(Xnorm[:,2])
max3 = max(Xnorm[:,2])
print "var1_TRUE"
print min(X[:,0])
print max(X[:,0])
print "var1_NORM"
print min(Xnorm[:, 0])
print max(Xnorm[:, 0])
print "var2"
print min(X[:, 1])
print max(X[:, 1])
print "var2_NORM"
print min(Xnorm[:, 1])
print max(Xnorm[:, 1])
print "var3"
print min(X[:, 2])
print max(X[:, 2])
print "var3_NORM"
print min(Xnorm[:, 2])
print max(Xnorm[:, 2])


# Run calibration
# -------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # create funcitionf of demand calaculation and send to theano

    # Call Gaussian process model as deterministic
    @as_op(itypes=[tt.dscalar,tt.dscalar,tt.dscalar], otypes=[tt.dvector])#,tt.dvector])
    def musamp(var1,var2,var3):
        mufunc=np.empty(1)
        sigmafunc=np.empty(1)
        # You can use the GP model to provide you either the mean predicted value, or mean+stdev of a set of inputs.
        # I am currently ignoring the GP error term (stdev) in teh calibration, as it seems computationally much more efficient
        # to do so.
        #mufunc[0],sigmafunc[0] = gp.predict([[var1,var2,var3]],return_std=True)
        mufunc[0] = gp.predict([[var1, var2, var3]], return_std=False)
        return mufunc#, sigmafunc

    with pm.Model() as basic_model:

        # get priors for the input variables of the model assuming they are all uniform
        # priors = [pm.Uniform(name, lower=a, upper=b) for name, a, b in zip(var_names, pdf_arg['min'].values,
        #                                                                   pdf_arg['max'].values)]
        # Set priors on variable data
        var1 = pm.Uniform('var1', lower=min1, upper=max1)
        var2 = pm.Uniform('var2', lower=min2, upper=max2)
        var3 = pm.Uniform('var3', lower=min3, upper=max3)

        # get priors for the model inadequacy and the measurement errors.
        # phi = pm.Uniform('phi', lower=0, upper=0.01)
        # err = pm.Uniform('err', lower=0, upper=0.02)
        sigma = pm.HalfNormal('sigma', sd=10) # Single error term as there is no measurement error due to analytical model

        # Call sample from GPmodel
        #mu,errSD = musamp(var1,var2,var3) # option when one wants to include GPmodel stdev as error term
        mu = musamp(var1, var2, var3) # option when ignoring GPmodel error

        # Likelihood (sampling distribution) of observations
        #
        y_obs = pm.Normal('y_obs', mu=mu, sd=(sigma), observed=0) # Note, how there is no observed data

    #basic_model.logp({'y': 0.})


    with basic_model:
        print "Starting ...."
        start = pm.find_MAP(fmin=optimize.fmin_powell)#=optimize.fmin_powell)
        print "Assigning step method...."
        step = pm.Metropolis(state=start)
        print "Running sample algorightm...."
        trace = pm.sample(20000, tune=200, step=step, start=start,njobs=1)#tune=100
        pm.backends.text.dump(os.getcwd(), trace)
        pm.traceplot(trace)
        plt.show()


