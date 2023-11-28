import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../..')
from HeteroscedasticGP import*
from GP import *
from EMGP0 import ExpectationMaximisationGaussianProcess as EM_GP0
from sklearn.metrics import mean_squared_error

plt.close('all')

# Read excel file
file_name = 'Synthetic.xlsx'
df = pd.read_excel(file_name, sheet_name='Training')
xNewdf = pd.read_excel(file_name, sheet_name='Testing')
labels_df = pd.read_excel(file_name, sheet_name='Real labels')

# Get training data
X = df['X'].values
Y = df['Y'].values
N = len(Y)
xNew = xNewdf['X_star']

# Get real labels
c0 = labels_df['Noise0'].values
c1 = labels_df['Noise1']
c2 = labels_df['Noise2']
not_nan = ~np.isnan(labels_df['Noise1'].values)
c1 = c1[not_nan]
c1 = [int(i) for i in c1]
not_nan = ~np.isnan(labels_df['Noise2'].values)
c2 = c2[not_nan]
c2 = [int(i) for i in c2]
indices = [c0, c1, c2]
    

### Create and train standard Gaussian Processes (to compare to)
gp = GP(X, Y, N, initial_hyperparameters=[1, 0.5],
        hyperparameter_bounds=((1e-6,None),(1e-6,2)),
        normalise_y=True)
gp.train()
muGP, stdGP = gp.predict(xNew, 2*N)

### Create and train a Heteroscedastic GP (to compare to)
hgp = HeteroscedasticGP(X, Y, N, initial_hyperparameters=[1, 0.5],
                        hyperparameter_bounds=((1e-6,None),(1e-6,2)),
                        gpz_hyperparameter_bounds=((1, None),(1e-6, 100)),
                        normalise_y=True)
hgp.train()
muh, stdh = hgp.predict(xNew, 2*N)

### Create and train EEEM GPs
mixGP = EM_GP0(X, Y, N, 3, initial_sigmas=[38, 135, 173], initial_hyperparameters=[1, 0.5],
              hyperparameter_bounds=((1e-6,10),(1e-6,2)), normalise_y=True,
              plot_conv=True)
mixGP.train()
muMix, stdMix = mixGP.predict(xNew, 2*N)


### CALCULATING THE OVERALL MSE
F = 150 * xNew * np.sin(xNew)
print("Mean Squared Error (GP)   : ", mean_squared_error(muGP, F))
print("Mean Squared Error (HGP): ", mean_squared_error(muh, F))
print("Mean Squared Error (EM-GP)  : ", mean_squared_error(muMix, F))


### Print results for the EM-GP model
print('\n MODEL PARAMETERS EM-GP (with normalisation): \n')
print(' Number of components identified, K = ', mixGP.noise_sources)
print('Proportionalities: ', mixGP.pies)
print('Noise Stds: ', mixGP.stds)
print('Hyperparameters: ', mixGP.hyperparameters)

############################## PLOT THE RESULST ##############################
color_iter = ['lightgreen', 'orange', 'red']
enumerate_real = [i for i in range(3)]
enumerate_K = [i for i in range(mixGP.noise_sources)]

############################ REAL NOISE LABELS ###############################

plt.figure()
plt.title(" Data corrupted with non-Gaussian noise ", fontsize=20)
plt.plot(xNew, F, color="black", label="Sine function")
for i, (k, c) in enumerate(zip(enumerate_real, color_iter)):
    plt.plot(X[indices[k]], Y[indices[k]], 'o',color=c, markersize = 9,
             label='Gaussian noise '+str(k+1))
plt.xlabel(' X ', fontsize=20)
plt.ylabel(' f(X) ', fontsize=20)
plt.legend(loc=3, prop={"size":25})

############################ Clustering miss-classified #######################
if mixGP.noise_sources == 3:
    missing = [np.setdiff1d(indices[0], mixGP.indices[0]),
               np.setdiff1d(indices[1], mixGP.indices[1]),
               np.setdiff1d(indices[2], mixGP.indices[2])]
    plt.figure()
    plt.title(" Miss-classified data ", fontsize=20)
    plt.plot(xNew, F, color="black", label="Sine function")
    plt.plot(X, Y, 'o', color="black", label="Observations")
    for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
        plt.plot(X[missing[k]], Y[missing[k]], 'kx',color=c,
                 markersize = 10, label='Noise Level '+str(k+1))
    plt.xlabel(' X ', fontsize=16)
    plt.ylabel(' f(X) ', fontsize=16)
    plt.legend(loc=3, prop={"size":20})
    plt.show()
    missed = max(len(missing[0]), len(missing[1]), len(missing[2]))
    print('\n The number of miss-clustered observations is: ', missed,
          '\n Correclty identified: ', N - missed,
          '\n with a clustering accuracy of: ', ((N - missed)*100)/150)
else:
    print("Estimated K different from the noise source")

plt.figure()
plt.title(" Clustering performance ", fontsize=20)
color_iter = ['black', 'red', 'lightgreen']
nl = ['Noise level 2', 'Noise level 1', 'Noise level 0']
plt.plot(xNew, muMix, color="green", label="EM-GP")
for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
    plt.plot(X[mixGP.indices[k]], Y[mixGP.indices[k]], 'o',color=c,
             markersize = 9, label=nl[k])
plt.xlabel(' X ', fontsize=16)
plt.ylabel(' f(X) ', fontsize=16)
plt.legend(loc=0, prop={"size":20})
plt.show()


### Regression performance: Comparison with a HGP
plt.figure()
plt.plot(xNew, F, color='black', linewidth = 4, label='Sine function')
plt.plot(xNew, muGP, color='blue', linestyle = '-.', linewidth = 4,
         label='GP')
plt.plot(xNew, muh, color='red', linestyle = '--', linewidth = 4, label='VHGP')
plt.plot(xNew, muMix, color='orange', linestyle=':', linewidth = 4,
         label='EM-GP')
plt.title('Regression Performance', fontsize=20)
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

# Plot results
plt.figure()
plt.fill_between(xNew, muMix[:,0] + 3*stdMix[:,0],
                 muMix[:,0] - 3*stdMix[:,0],
                 alpha=0.5,color='lightgreen',
                 label='Confidence \nBounds (EM-GP)')
plt.plot(X, Y, 'o', color='black')
plt.plot(xNew, muMix, linewidth=2.5, color='green', label='EM-GP')
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

plt.show()