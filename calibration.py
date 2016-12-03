"""
===========================
Bayesian calibration routine

based on work of (Patil_et_al, 2010 - #http://www.map.ox.ac.uk/media/PDF/Patil_et_al_2010.pdf) in MCMC in pyMC3
and the works of bayesian calibration of (Kennedy and O'Hagan, 2001)
===========================
J. Fonseca  script development          27.10.16


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

# import cea.globalvar as gv
# gv = gv.GlobalVariables()
# import cea.inputlocator as inputlocator
# scenario_path = gv.scenario_reference
# locator = inputlocator.InputLocator(scenario_path=scenario_path)
# weather_path = locator.get_default_weather()


__author__ = "Jimeno A. Fonseca"
__copyright__ = "Copyright 2016, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Jimeno A. Fonseca"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Daren Thomas"
__email__ = "cea@arch.ethz.ch"
__status__ = "Production"


def calibration_main(group_var, building_name, building_load, retrieve_results, niter):
    # import arguments of probability density functions (PDF) of variables and create priors:
    # pdf_arg = pd.concat([pd.read_excel(locator.get_uncertainty_db(),
    #                                   group, axis=1) for group in group_var]).set_index('name')
    # var_names = pdf_arg.index

    # import measured data for building and building load:
    # obs_data = pd.read_csv(locator.get_demand_measured_file(building_name))[building_load].values
    #
    global gp
    gp = joblib.load('SinusGP.pkl')
    obs_data = []
    for x in range(2001):
        obs_data.append(0)
    print len(obs_data)

    xl = pd.ExcelFile("SinusTest.xlsx")
    print xl.sheet_names
    df = xl.parse("Data")
    print df.head()
    obs_data = df.iloc[0:2002, 6].values
    X = df.iloc[0:2002, 0:3].values
    Xnorm = preprocessing.normalize(X)
    y = df.iloc[0:2002, 6].values
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


    # create funcitionf of demand calaculation and send to theano

    @theano.compile.ops.as_op(itypes=[tt.dscalar,tt.dscalar,tt.dscalar], otypes=[tt.dvector])
    def mu(var1, var2, var3):
        mu=np.empty(1)
        sigma=np.empty(1)
        mu[0] = gp.predict([[var1,var2,var3]],return_std=False)
        return mu

    @theano.compile.ops.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar], otypes=[tt.dvector])
    def sigma2(var1, var2, var3):
        sigma = np.empty(1)
        mutemp,sigmatemp = gp.predict([[var1, var2, var3]], return_std=True)
        sigma[0] = sigmatemp
        return sigma


    with pm.Model() as basic_model:

        # get priors for the input variables of the model assuming they are all uniform
        # priors = [pm.Uniform(name, lower=a, upper=b) for name, a, b in zip(var_names, pdf_arg['min'].values,
        #                                                                   pdf_arg['max'].values)]

        #u_win = pm.Uniform('u_win', lower=0.9, upper=3.1)
        #u_wall = pm.Uniform('u_wall', lower=0.11, upper=1.5)
        var1 = pm.Uniform('var1', lower=min1, upper=max1)
        var2 = pm.Uniform('var2', lower=min2, upper=max2)
        var3 = pm.Uniform('var3', lower=min3, upper=max3)
        # get priors for the model inadequacy and the measurement errors.
        # phi = pm.Uniform('phi', lower=0, upper=0.01)
        # err = pm.Uniform('err', lower=0, upper=0.02)
        sigma = pm.HalfNormal('sigma', sd=100)

        ### NEW FOR COUPLED CALIBRATION TO DLM-GASP

        #mu, sigmaEMU = callGPmodel(var1,var2,var3)
        #muEMU, sigmaEMU = gp.predict([var1, var2, var3], return_std=True)

        mu = mu(var1,var2,var3)
        sigma2 = sigma2(var1,var2,var3)
        #sigma = pm.Deterministic('sigma', sigmaEMU)

        ### END OF NEW FOR COUPLED CALIBRATION WITH DLM-GASP

        # Likelihood (sampling distribution) of observations
        #
        y_obs = pm.Normal('y_obs', mu=mu, sd=(sigma+sigma2), observed=0)
        #y = pm.DensityDist('y', logp(var1,var2,var3))

    #basic_model.logp({'y': 0.})

    if retrieve_results:
        o=0
        #with basic_model:
            # trace = pm.backends.text.load(locator.get_calibration_folder())
            # pm.traceplot(trace)
            # plt.show()
    else:
        with basic_model:
            start = pm.find_MAP()#=optimize.fmin_powell)
            #C = approx_hessian(model.test_point)
            step = pm.Metropolis(state=start)
            trace = pm.sample(2000, step, init=start)
            pm.backends.text.dump(os.getcwd(), trace)
            pm.traceplot(trace)
            plt.show()
    return


def run_as_script():
    group_var = ['THERMAL']
    building_name = 'B01'
    building_load = 'Qhsf_kWh'
    retrieve_results = False  # flag to retrieve and analyze results from calibration
    calibration_main(group_var, building_name, building_load, retrieve_results, niter=100)


if __name__ == '__main__':
    run_as_script()