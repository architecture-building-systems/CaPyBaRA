import numpy as np
import matplotlib.pyplot as plt
import math as m
from tkFileDialog import askdirectory
import os
import pandas as pd
from scipy.interpolate import spline
from scipy import stats

ymodel = []
ydata = []

Nsamples = 400
# Open file browser to search for director --------------------------------
def GetFileDirectory(str1):
    directory = askdirectory(title=str1) # show an "Open" dialog box and return the path to the selected file
    print(directory)
    if directory == "":
        exit()
    return directory;

maindir = GetFileDirectory("Find main simulation folder for plotting")
subdirs = os.listdir(maindir)
averages = []

for i in range(1,Nsamples+1):
    subdirpath = os.path.join(maindir,str(i))
    if os.path.isdir(subdirpath):
        print i
        filepath = os.path.join(subdirpath,"Output_XXX.csv")
        data1 = pd.read_csv(filepath)
        if i==1:
            data5 = data1.iloc[168:734, 7:11].values
            data6 = np.sum(data5, axis=1)
        data2 = data1.iloc[168:734,2:6].values
        data3 = np.sum(data2, axis=1)
        ymodel.extend(data3.tolist())
        ydata.extend(data6.tolist())
        for j in range(0,24):
            value = 0
            for k in range(0,23):
                value += data3[(k-1)*24+j]
            averages.append(value)
        #plt.plot(averages,color='k', alpha=0.04, lw=0.5)
        del(averages[:])
        plt.plot(data3, color='k', alpha=0.04, lw=0.5)

mean = np.mean(ydata)
SS = []
SSres = []
for i in range(0,len(ymodel)):
    SSval = (ydata[i]-mean)**2
    SSresval = (ydata[i]-ymodel[i])**2
    SS.append(SSval)
    SSres.append(SSresval)

r_squared = 1-np.sum(SSres)/np.sum(SS)
print len(SS)/20
print len(ymodel)

print r_squared

r_square_persim =[]
start = 0
for i in range(0,Nsamples):
    end = start + 566
    r_square_persim.append(1-np.sum(SSres[start:end])/np.sum(SS[start:end]))
    start = start + 566

newR_square_persim = sorted(r_square_persim)
#print newR_square_persim
#plt.scatter(ymodel,ydata,s=0.5)
X = [0,40000]
Y = [0,40000]
#plt.plot(X,Y)
#plt.axis((0,25000,0,25000))
#plt.plot(newR_square_persim)
#plt.show()
#x_ax = data1.iloc[168:734,0].values
#xn_ax = np.linspace(x_ax.min(),x_ax.max(),566*10)
data2 = data1.iloc[168:734,7:11].values
data3 = np.sum(data2, axis=1)
plt.plot(data3, color='b', alpha=1, lw=2)
#data4 = spline(x_ax, data3, xn_ax)
#plt.plot(xn_ax,data4, color='k', alpha=0.02, lw=0.5)
#slope, intercept, r_value, p_value, std_err = stats.linregress(ydata, ymodel)
#r_squared = r_value**2
#print r_squared

for j in range(0, 24):
    value = 0
    for k in range(0, 23):
        value += data3[(k - 1) * 24 + j]
    averages.append(value)
#plt.plot(averages, color='b', alpha=1, lw=2)
del (averages[:])

plt.show()

#for child in os.listdir(maindir):
#    test_path = os.path.join(maindir, child)
#    if os.path.isdir(test_path):
#        print test_path


#    plt.plot(datastore,color='k',alpha=0.02,lw=0.5)
#plt.show()
