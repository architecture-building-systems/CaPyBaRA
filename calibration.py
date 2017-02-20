"""
===========================
Bayesian calibration routine

based on work of (Patil_et_al, 2010 - #http://www.map.ox.ac.uk/media/PDF/Patil_et_al_2010.pdf) in MCMC in pyMC3
and the works of bayesian calibration of (Kennedy and O'Hagan, 2001)

===========================
A. Rysanek, J. Fonseca  20.02.17


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

#__author__ = "Jimeno A. Fonseca"
#__copyright__ = "Copyright 2016, Architecture and Building Systems - ETH Zurich"
#__credits__ = ["Jimeno A. Fonseca"]
#__license__ = "MIT"
#__version__ = "0.1"
#__maintainer__ = "Daren Thomas"
#__email__ = "cea@arch.ethz.ch"
#__status__ = "Production"

gp = joblib.load('TRNSYSgp.pkl')
obs_data = []
for x in range(2001):
    obs_data.append(0)
#print len(obs_data)


xl = pd.ExcelFile("OutputSummary_17-01-12-202726.xlsx")
print xl.sheet_names
df = xl.parse("OutputSummary_17-01-12-202726")
print df.head()
df = pd.read_csv("OutputSummary_17-01-12-202726.csv")

X = df.iloc[0:4999,1:35].values
Xnorm = preprocessing.normalize(X)
min_max_scaler = preprocessing.MinMaxScaler()
Xnorm = min_max_scaler.fit_transform(X)
y = df.iloc[0:4999,47].values

#xl = pd.ExcelFile("SinusTest.xlsx")
#print xl.sheet_names
#df = xl.parse("Data")
#print df.head()
#obs_data = df.iloc[0:2002, 6].values
#X = df.iloc[0:2002, 0:3].values
#Xnorm = preprocessing.normalize(X)
#y = df.iloc[0:2002, 6].values

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


if __name__ == '__main__':
    # create funcitionf of demand calaculation and send to theano

    @as_op(itypes=[tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar], otypes=[tt.dvector])
    def musamp(var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14,var15,var16,var17,var18,var19,var20,var21,var22,var23,var24,var25,var26,var27,var28,var29,var30,var31,var32,var33,var34,var35):
        mufunc=np.empty(1)
        sigmafunc=np.empty(1)
        #var22 = 0.05
        #var23 = 0.41
        #var24 = 0.45
        #var25 = 0.30
        #var26 = 0.73
        #var27 = 0.79
        #var28 = 0.663
        #var29 = 0.40
        #var30 = 0.248
        #var31 = 0.172
        #var32 = 0.36
        #var33 = 0.406
        #var34 = 0.55
        #var35 = 0.5
        mufunc[0] = gp.predict([[var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14,var15,var16,var17,var18,var19,var20,var21,var22,var23,var24,var25,var26,var27,var28,var29,var30,var31,var32,var33,var34,var35]],return_std=False)

        musamp.grad = lambda *x: x[0]        #mufunc[0]=500
        #return mufunc, sigmafunc
        return mufunc

    with pm.Model() as basic_model:

        # get priors for the input variables of the model assuming they are all uniform
        # priors = [pm.Normal(name, lower=a, upper=b) for name, a, b in zip(var_names, pdf_arg['min'].values,
        #                                                                   pdf_arg['max'].values)]

        #u_win = pm.Normal('u_win', lower=0.9, upper=3.1)
        #u_wall = pm.Normal('u_wall', lower=0.11, upper=1.5)
        #var1 = pm.Normal('var1', lower=min1, upper=max1)
        #var2 = pm.Normal('var2', lower=min2, upper=max2)
        #var3 = pm.Normal('var3', lower=min3, upper=max3)

        ###var1 = pm.Normal('cp_HRM',lower=-0.2,upper=1)
        ###var2 = pm.Normal('cp_HRO1', lower=-0.2, upper=1)
        ###var3 = pm.Normal('cp_HRO2', lower=-0.2, upper=1)
        ###var4 = pm.Normal('cp_HS', lower=-0.2, upper=1)
        ###var5 = pm.Normal('cp_HRD', lower=-0.2, upper=1)
        ###var6 = pm.Normal('x_BNDCEIL', lower=0, upper=2)
        ###var7 = pm.Normal('GRD_REF', lower=-1, upper=1)
        ###var8 = pm.Normal('ACCPAN_ABS', lower=0, upper=1)
        ###var9 = pm.Normal('ACCPAN_INSUL', lower=-0.5, upper=1)
        ###var10 = pm.Normal('x_BNDLOORINSUL', lower=0, upper=1)
        ###var11 = pm.Normal('INFIL_ACH', lower=0, upper=2)
        ###var12 = pm.Normal('x_HRMWALL', lower=0, upper=2)
        ###var13 = pm.Normal('WFR', lower=-0.5, upper=1)
        ###var14 = pm.Normal('EQscale_HRM', lower=-0.5, upper=1)
        ###var15 = pm.Normal('EQscale_HRO1', lower=0, upper=1)
        ###var16 = pm.Normal('EQscale_HRO2', lower=0, upper=2)
        ###var17 = pm.Normal('EQscale_HS', lower=-1, upper=1)
        ###var18 = pm.Normal('EQscale_HRD', lower=0, upper=1)
        ###var19 = pm.Normal('Tout_err', lower=0, upper=1)
        ###var20 = pm.Normal('Solarout_err', lower=-1, upper=1)
        ###var21 = pm.Normal('Floor4Tadj', lower=0, upper=2)

        var1 = pm.Triangular('cp_HRM',lower=0,c=0.5,upper=1)
        var2 = pm.Triangular('cp_HRO1',lower=0,c=0.5,upper=1)
        var3 = pm.Triangular('cp_HRO2', lower=0,c=0.5,upper=1)
        var4 = pm.Triangular('cp_HS',lower=0,c=0.5,upper=1)
        var5 = pm.Triangular('cp_HRD', lower=0,c=0.5,upper=1)
        var6 = pm.Triangular('x_BNDCEIL', lower=0,c=0.5,upper=1)
        var7 = pm.Triangular('GRD_REF', lower=0,c=0.5,upper=1)
        var8 = pm.Triangular('ACCPAN_ABS', lower=0,c=0.5,upper=1)
        var9 = pm.Triangular('ACCPAN_INSUL', lower=0,c=0.5,upper=1)
        var10 = pm.Triangular('x_BNDLOORINSUL', lower=0,c=0.5,upper=1)
        var11 = pm.Triangular('INFIL_ACH', lower=0,c=0.5,upper=1)
        var12 = pm.Triangular('x_HRMWALL', lower=0,c=0.5,upper=1)
        var13 = pm.Triangular('WFR', lower=0,c=0.5,upper=1)
        var14 = pm.Triangular('EQscale_HRM', lower=0,c=0.5,upper=1)
        var15 = pm.Triangular('EQscale_HRO1', lower=0,c=0.5,upper=1)
        var16 = pm.Triangular('EQscale_HRO2', lower=0,c=0.5,upper=1)
        var17 = pm.Triangular('EQscale_HS', lower=0,c=0.5,upper=1)
        var18 = pm.Triangular('EQscale_HRD', lower=0,c=0.5,upper=1)
        var19 = pm.Triangular('Tout_err', lower=0,c=0.5,upper=1)
        var20 = pm.Triangular('Solarout_err', lower=0,c=0.5,upper=1)
        var21 = pm.Triangular('Floor4Tadj', lower=0,c=0.5,upper=1)
        var22 = pm.Triangular('EQadj_6am', lower=0,c=0.5,upper=1)
        var23 = pm.Triangular('EQadj_7am', lower=0,c=0.5,upper=1)
        var24 = pm.Triangular('EQadj_8am', lower=0,c=0.5,upper=1)
        var25 = pm.Triangular('EQadj_9am', lower=0,c=0.5,upper=1)
        var26 = pm.Triangular('EQadj_10am', lower=0,c=0.5,upper=1)
        var27 = pm.Triangular('EQadj_11am', lower=0,c=0.5,upper=1)
        var28 = pm.Triangular('EQadj_12pm', lower=0,c=0.5,upper=1)
        var29 = pm.Triangular('EQadj_1pm', lower=0,c=0.5,upper=1)
        var30 = pm.Triangular('EQadj_2pm', lower=0,c=0.5,upper=1)
        var31 = pm.Triangular('EQadj_3pm', lower=0,c=0.5,upper=1)
        var32 = pm.Triangular('EQadj_4pm', lower=0,c=0.5,upper=1)
        var33 = pm.Triangular('EQadj_5pm', lower=0,c=0.5,upper=1)
        var34 = pm.Triangular('EQadj_6pm', lower=0,c=0.5,upper=1)
        var35 = pm.Triangular('EQadj_7pm', lower=0,c=0.5,upper=1)
        
        # var1 = pm.Lognormal('cp_HRM',mu=-0.51,tau=5)
        # var2 = pm.Lognormal('cp_HRO1',mu=-0.51,tau=5)
        # var3 = pm.Lognormal('cp_HRO2', mu=-0.51,tau=5)
        # var4 = pm.Lognormal('cp_HS',mu=-0.51,tau=5)
        # var5 = pm.Lognormal('cp_HRD', mu=-0.51,tau=5)
        # var6 = pm.Normal('x_BNDCEIL', mu=0.5,sd=0.17)
        # var7 = pm.Normal('GRD_REF', mu=0.5,sd=0.17)
        # var8 = pm.Normal('ACCPAN_ABS', mu=0.5,sd=0.17)
        # var9 = pm.Normal('ACCPAN_INSUL', mu=0.5,sd=0.17)
        # var10 = pm.Normal('x_BNDLOORINSUL', mu=0.5,sd=0.17)
        # var11 = pm.Normal('INFIL_ACH', mu=0.5,sd=0.17)
        # var12 = pm.Normal('x_HRMWALL', mu=0.5,sd=0.17)
        # var13 = pm.Normal('WFR', mu=0.5,sd=0.17)
        # var14 = pm.Normal('EQscale_HRM', mu=0.5,sd=0.17)
        # var15 = pm.Normal('EQscale_HRO1', mu=0.5,sd=0.17)
        # var16 = pm.Normal('EQscale_HRO2', mu=0.5,sd=0.17)
        # var17 = pm.Normal('EQscale_HS', mu=0.5,sd=0.17)
        # var18 = pm.Normal('EQscale_HRD', mu=0.5,sd=0.17)
        # var19 = pm.Normal('Tout_err', mu=0.5,sd=0.17)
        # var20 = pm.Normal('Solarout_err', mu=0.5,sd=0.17)
        # var21 = pm.Normal('Floor4Tadj', mu=0.5,sd=0.17)
        # var22 = pm.Normal('EQadj_6am', mu=0.5,sd=0.17)
        # var23 = pm.Normal('EQadj_7am', mu=0.5,sd=0.17)
        # var24 = pm.Normal('EQadj_8am', mu=0.5,sd=0.17)
        # var25 = pm.Normal('EQadj_9am', mu=0.5,sd=0.17)
        # var26 = pm.Normal('EQadj_10am', mu=0.5,sd=0.17)
        # var27 = pm.Normal('EQadj_11am', mu=0.5,sd=0.17)
        # var28 = pm.Normal('EQadj_12pm', mu=0.5,sd=0.17)
        # var29 = pm.Normal('EQadj_1pm', mu=0.5,sd=0.17)
        # var30 = pm.Normal('EQadj_2pm', mu=0.5,sd=0.17)
        # var31 = pm.Normal('EQadj_3pm', mu=0.5,sd=0.17)
        # var32 = pm.Normal('EQadj_4pm', mu=0.5,sd=0.17)
        # var33 = pm.Normal('EQadj_5pm', mu=0.5,sd=0.17)
        # var34 = pm.Normal('EQadj_6pm', mu=0.5,sd=0.17)
        # var35 = pm.Normal('EQadj_7pm', mu=0.5,sd=0.17)

        #var35 = 0.51

        # get priors for the model inadequacy and the measurement errors.
        # phi = pm.Normal('phi', lower=0, upper=0.01)
        # err = pm.Normal('err', lower=0, upper=0.02)
        sigma = pm.HalfNormal('sigma', sd=0.01)
        #sigma = pm.Normal('sigma',mu=-750, sd=10)

        ### NEW FOR COUPLED CALIBRATION TO DLM-GASP

        #mu, sigmaEMU = callGPmodel(var1,var2,var3)
        #muEMU, sigmaEMU = gp.predict([var1, var2, var3], return_std=True)

        #muGP = musamp(var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14,var15,var16,var17,var18,var19,var20,var21)
        mu = pm.Deterministic('mu',musamp(var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14,var15,var16,var17,var18,var19,var20,var21,var22,var23,var24,var25,var26,var27,var28,var29,var30,var31,var32,var33,var34,var35))
        #err = pm.HalfNormal('err', sd=errSD[0])
        #mu = pm.Normal('mu',mu=muGP[0],sd=errSD[0])

        ### END OF NEW FOR COUPLED CALIBRATION WITH DLM-GASP

        # Likelihood (sampling distribution) of observations
        #
        #y_obs = pm.Normal('y_obs', mu=muGP[0], sd=sigma, observed=0)
        #y_obs = pm.Normal('y_obs', mu=mu, sd=sigma, observed=0)
        y_obs = pm.Normal('y_obs', mu=mu,sd=sigma, observed=0.15)

        #y = pm.DensityDist('y', logp(var1,var2,var3))

    #basic_model.logp({'y': 0.})


    with basic_model:
        print "Starting ...."
        start = pm.find_MAP(fmin=optimize.fmin_powell)#=optimize.fmin_powell)
        #start = {'var1': 0.5, 'var2': 0.5, 'var3': 0.5, 'var4': 0.5, 'var5': 0.5, 'var6': 0.5, 'var7': 0.5, 'var8': 0.5, 'var9': 0.5, 'var10': 0.5, 'var11': 0.5, 'var12': 0.5, 'var13': 0.5, 'var14': 0.5, 'var15': 0.5, 'var16': 0.5, 'var17': 0.5, 'var18': 0.5, 'var19': 0.5, 'var20': 0.5, 'var21': 0.5, 'sigma': 4000}
        #C = approx_hessian(model.test_point)
        #step = pm.HamiltonianMC([var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14,var15,var16,var17,var18,var19,var20,var21,sigma,err])
        print "Assigning step method...."
        #step1 = pm.Metropolis(vars=[var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14,var15,var16,var17,var18,var19,var20,var21])
        #step2 = pm.Metropolis(vars=[muGP,sigma])
        step = pm.Metropolis()
        #step = pm.NUTS()
        print "Running sample algorightm...."
        trace = pm.sample(5000, tune=500, step = step, njobs=1)#tune=100
        pm.traceplot(trace)
        pm.backends.text.dump(os.getcwd(), trace)
        plt.show()


