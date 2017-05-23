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
from math import sin,pi
import glob
import csv
from xlsxwriter.workbook import Workbook

#__author__ = "Jimeno A. Fonseca"
#__copyright__ = "Copyright 2016, Architecture and Building Systems - ETH Zurich"
#__credits__ = ["Jimeno A. Fonseca"]
#__license__ = "MIT"
#__version__ = "0.1"
#__maintainer__ = "Daren Thomas"
#__email__ = "cea@arch.ethz.ch"
#__status__ = "Production"

gp = joblib.load('GPR_TestCase_1.pkl')
obs_data = []
for x in range(2001):
    obs_data.append(0)
#print len(obs_data)

Xtest = np.empty(7)


Xtest[0] =(100)
Xtest[1] =(-6)
Xtest[2] =(-4)
Xtest[3] =(100)
Xtest[4] =(500)
Xtest[5] =(150)
#Xtest[6] =(12)
Xtest[6] =(0)
print Xtest[:]

y_pred, y_std = gp.predict(Xtest[:], return_std=True)
print y_pred
print y_std

xl = pd.ExcelFile("GPRdataset.xls")
print xl.sheet_names
df = xl.parse("Sheet 1")
print df.head()
#df = pd.read_csv("GPR_TestCase_1.csv")

X = df.iloc[0:9999,1:8].values
Xnorm = preprocessing.normalize(X)
min_max_scaler = preprocessing.MinMaxScaler()
Xnorm = min_max_scaler.fit_transform(X)
y = df.iloc[0:9999,8].values

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

Nsteps = 600

A = 100
B = -6
C = -4
D = 100
E = 500
F = 150
G = 12
H = 0

Niter = 10000
Nsteps = 600
xRange = list(range(Nsteps))

workbook = Workbook('Posteriors.xlsx')

# Generate random list
err = np.random.uniform(0,50,Nsteps)

# Generate *real* dataset
zdata = []
for x in range (0,Nsteps):
    zdata.append(A*(sin(pi*(B+x)/G+H)+sin((C+x))*pi/G)+D*sin(pi*x/E)+F+np.random.normal(0,20))

for case in xrange(6):
    if __name__ == '__main__':
        # create funcitionf of demand calaculation and send to theano

        @as_op(itypes=[tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar], otypes=[tt.dvector])
        def musamp(var1,var2,var3,var4,var5,var6,var7):
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

            #mufunc[0] = gp.predict([[var1,var2,var3,var4,var5,var6,var7]],return_std=False)

            # Initialize MSE
            mse = []

            # Assess Data
            z = []
            for x in range(0, Nsteps):
                z.append(var1 * (sin(pi * (var2 + x) / 12 + var7) + sin((var3 + x)) * pi /12) + var4 * sin(pi * x / var5) + var6)
                mse.append((z[x] - zdata[x]) ** 2)
            mufunc[0] = ((sum(mse) / Nsteps) ** 0.5)

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

            #var1 = pm.Triangular('A',lower=0,c=0.5,upper=1)
            #var2 = pm.Triangular('B',lower=0,c=0.5,upper=1)
            #var3 = pm.Triangular('C', lower=0,c=0.5,upper=1)
            #var4 = pm.Triangular('D',lower=0,c=0.5,upper=1)
            #var5 = pm.Triangular('E', lower=0,c=0.5,upper=1)
            #var6 = pm.Triangular('F', lower=0,c=0.5,upper=1)
            #var7 = pm.Triangular('H', lower=0,c=0.5,upper=1)

            #var1 = pm.Uniform('A', lower=0,upper=50)
            #var2 = pm.Uniform('B', lower=0,upper=6)
            #var3 = pm.Uniform('C', lower=0,upper=4)
            #var4 = pm.Uniform('D', lower=150,upper=200)
            #var5 = pm.Uniform('E', lower=-300,upper=400)
            #var6 = pm.Uniform('F', lower=-100,upper=120)
            #var7 = pm.Uniform('H', lower=-1,upper=-0.3)

            if case == 0:
                var1 = pm.Normal('A',mu=100,sd=10)
                var2 = pm.Normal('B',mu=-6,sd=0.6)
                var3 = pm.Normal('C',mu=-4,sd=0.4)
                var4 = pm.Normal('D',mu=100,sd=10)
                var5 = pm.Normal('E',mu=500,sd=50)
                var6 = pm.Normal('F',mu=150,sd=15)
                var7 = pm.Normal('H',mu=0,sd=0.05)
            elif case == 1:
                var1 = pm.Normal('A', mu=100, sd=25)
                var2 = pm.Normal('B', mu=-6, sd=1.5)
                var3 = pm.Normal('C', mu=-4, sd=0.8)
                var4 = pm.Normal('D', mu=100, sd=25)
                var5 = pm.Normal('E', mu=500, sd=75)
                var6 = pm.Normal('F', mu=150, sd=40)
                var7 = pm.Normal('H', mu=0, sd=0.1)
            elif case == 2:
                var1 = pm.Normal('A', mu=100, sd=50)
                var2 = pm.Normal('B', mu=-6, sd=3)
                var3 = pm.Normal('C', mu=-4, sd=2)
                var4 = pm.Normal('D', mu=100, sd=50)
                var5 = pm.Normal('E', mu=500, sd=100)
                var6 = pm.Normal('F', mu=150, sd=80)
                var7 = pm.Normal('H', mu=0, sd=0.5)
            elif case == 3:
                var1 = pm.Normal('A', mu=120, sd=30)
                var2 = pm.Normal('B', mu=-3.5, sd=2.5)
                var3 = pm.Normal('C', mu=5, sd=1)
                var4 = pm.Normal('D', mu=30, sd=30)
                var5 = pm.Normal('E', mu=150, sd=90)
                var6 = pm.Normal('F', mu=100, sd=50)
                var7 = pm.Normal('H', mu=-0.25, sd=0.2)
            elif case == 4:
                var1 = pm.Normal('A', mu=25, sd=15)
                var2 = pm.Normal('B', mu=2, sd=1.5)
                var3 = pm.Normal('C', mu=2, sd=2)
                var4 = pm.Normal('D', mu=175, sd=15)
                var5 = pm.Normal('E', mu=350, sd=25)
                var6 = pm.Normal('F', mu=0, sd=30)
                var7 = pm.Normal('H', mu=-0.6, sd=0.3)
            elif case == 5:
                var1 = pm.Normal('A', mu=225, sd=50)
                var2 = pm.Normal('B', mu=0, sd=2)
                var3 = pm.Normal('C', mu=-7, sd=1.5)
                var4 = pm.Normal('D', mu=-50, sd=25)
                var5 = pm.Normal('E', mu=700, sd=200)
                var6 = pm.Normal('F', mu=300, sd=20)
                var7 = pm.Normal('H', mu=1, sd=0.4)


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
            sigma = pm.HalfNormal('sigma', sd=0.05)
            #sigma = pm.Normal('sigma',mu=-750, sd=10)

            ### NEW FOR COUPLED CALIBRATION TO DLM-GASP

            #mu, sigmaEMU = callGPmodel(var1,var2,var3)
            #muEMU, sigmaEMU = gp.predict([var1, var2, var3], return_std=True)

            #muGP = musamp(var1,var2,var3,var4,var5,var6,var7)
            mu = pm.Deterministic('mu',musamp(var1,var2,var3,var4,var5,var6,var7))
            #err = pm.HalfNormal('err', sd=errSD[0])
            #mu = pm.Normal('mu',mu=muGP[0],sd=errSD[0])

            ### END OF NEW FOR COUPLED CALIBRATION WITH DLM-GASP

            # Likelihood (sampling distribution) of observations
            #
            #y_obs = pm.Normal('y_obs', mu=muGP[0], sd=sigma, observed=0)
            y_obs = pm.Normal('y_obs', mu=mu, sd=sigma, observed=0)
            #y_obs = pm.Normal('y_obs', mu=mu,sd=sigma, observed=90)

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
            trace = pm.sample(7500, tune=500, step = step, njobs=1)#tune=100
            pm.traceplot(trace)
            pm.backends.text.dump(os.getcwd(), trace)

            for csvfile in glob.glob(os.path.join('.', 'chain-0.csv')):
                worksheet = workbook.add_worksheet('Case '+str(case))
                with open(csvfile, 'rb') as f:
                    reader = csv.reader(f)
                    for r, row in enumerate(reader):
                        for c, col in enumerate(row):
                            worksheet.write(r, c, col)
workbook.close()
            #plt.show()



