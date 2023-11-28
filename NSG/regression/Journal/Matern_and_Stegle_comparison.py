import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
sys.path.insert(0, 'C:\Diego\PhD\Code\phdCode')
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from EMGP import ExpectationMaximisationGaussianProcess as EMGP
from DPGP import DirichletProcessGaussianProcess as DPGP
from NSG import data_processing_methods as dpm
from sklearn.decomposition import PCA

"""
DPGP regression
"""

# ----------------------------------------------------------------------------
# USER OPTIONS
# ----------------------------------------------------------------------------

# Choose the scanner where the fault density data is read from
scanner = 'ISRA'

# Choose the furnace inputs
to_retain = ['10091 Furnace Load',
             '10271 C9 (T012) Upstream Refiner',
             '2922 Closed Bottom Temperature - Downstream Working End (PV)',
             '2921 Closed Bottom Temperature - Upstream Working End (PV)',
             '2918 Closed Bottom Temperature - Port 6 (PV)',
             '2923 Filling Pocket Closed Bottom Temperature Centre (PV)',
             '7546 Open Crown Temperature - Port 1 (PV)',
             '7746 Open Crown Temperature - Port 2 (PV)',
             '7522 Open Crown Temperature - Port 4 (PV)',
             '7483 Open Crown Temperature - Port 6 (PV)']

# ----------------------------------------------------------------------------
# LOAD DATA FOR TRANING AND TESTING
# ----------------------------------------------------------------------------

# Initialise empty data frames
X_df, X_df_test = pd.DataFrame(), pd.DataFrame()
Y_df, Y_df_test = pd.DataFrame(), pd.DataFrame()
Y_raw_df, Y_raw_df_test = pd.DataFrame(), pd.DataFrame()

# Loop over available files of post-processed data
for i in range(1, 5):
    file_name = ('NSG/Input Post-Processing ' + str(i) + ' ' +
                 scanner + '.xlsx')

    # The first 3 files are appended to become a single training data-frame
    if i < 4:
        X_df = X_df._append(pd.read_excel(file_name,
                                         sheet_name='input_data'))
        Y_df = Y_df._append(pd.read_excel(file_name,
                                         sheet_name='output_data'))
        Y_raw_df = Y_raw_df._append(pd.read_excel(file_name,
                                   sheet_name='raw_output_data'))

# Extract time lags from final file (should be the same for all)
T_df = pd.read_excel('NSG/Input Post-Processing 4 ISRA timelags.xlsx',
                      sheet_name='time_lags')

# Check data frames are the correct size and have the same column names
assert np.all(X_df.columns == T_df.columns)
assert len(X_df) == len(Y_df)
assert len(Y_df) == len(Y_raw_df)

# ----------------------------------------------------------------------------
# REMOVE INPUTS WE ARE NOT GOING TO USE
# ----------------------------------------------------------------------------

input_names = X_df.columns
for name in input_names:
    if name not in to_retain:
        X_df.drop(columns=name, inplace=True)
        T_df.drop(columns=name, inplace=True)

# Check that the data frames contain the correct number of inputs
assert len(X_df.columns) == len(to_retain)

# Check that the data frame input names match those in to_retain
assert set(X_df.columns) == set(to_retain)

# ----------------------------------------------------------------------------
# PRE-PROCESSING
# ----------------------------------------------------------------------------

# Finding fault density mean and std (at training points)
Y_mean = np.mean(Y_df['furnace_faults'].values)
Y_std = np.std(Y_df['furnace_faults'].values)

# Standardise training data
for i in range(np.shape(X_df)[1]):
    tag_name = X_df.columns[i]

    # Get the inputs statistics to use in the training and
    # testing data standardisation
    X_mean = np.mean(X_df.iloc[:, i])
    X_std = np.std(X_df.iloc[:, i])

    # Re-write X_df now with standardise data (at training points)
    X_df[tag_name] = dpm.standardise(X_df.iloc[:, i],
                                     X_mean,
                                     X_std)

# Standardise testing data
Y_df['furnace_faults'] = dpm.standardise(Y_df['furnace_faults'].values,
                                         Y_mean,
                                         Y_std)
# Process training data
X, Y, N, D, max_lag, time_lags = dpm.align_arrays(X_df, Y_df, T_df)

# Process raw target data in the same way as the post-processed
Y_raw = dpm.adjust_time_lag(Y_raw_df['raw_furnace_faults'].values,
                            shift=0, to_remove=max_lag)

# This just removes the first max_lag points from the date_time array.
Y_raw = dpm.adjust_time_lag(Y_raw_df['raw_furnace_faults'].values, shift=0,
                            to_remove=max_lag)

# Y_raw standardisation
Y_raw_mean = np.mean(Y_raw_df['raw_furnace_faults'].values)
Y_raw_std = np.std(Y_raw_df['raw_furnace_faults'].values)
Y_raw_stand = dpm.standardise(Y_raw, Y_raw_mean, Y_raw_std)

# This removes the first max_lag points from the date_time array.
date_time = dpm.adjust_time_lag(Y_df['Time stamp'].values, shift=0,
                                to_remove=max_lag)

# ----------------------------------------------------------------------------
# DPGP
# ----------------------------------------------------------------------------

# Select training dataset (N_max = 2600)
start = 4770
end = 5750

# Save memory: leave the X_df to identify the relevant inputs
del Y_df, Y_raw_df, T_df, dpm

# The training data
X_train = X[start: end]
Y_train = Y_raw_stand[start: end]
dt_train = date_time[start:end]
N_train = len(Y_train)
D = np.shape(X_train)[1]

# Length scales
ls = [7, 64, 7, 7.60, 7, 7, 7, 123, 76, 78]

ma = 1.0 * Matern(length_scale=ls, nu=2.5, length_scale_bounds=(0.05, 1e4))
# se = 1**2 * RBF(length_scale=ls, length_scale_bounds=(0.05, 1e4))
wn = WhiteKernel(noise_level=0.61**2, noise_level_bounds=(1e-5, 1))

kernel = ma + wn
del ma, wn, Y

# DPGP
dpgp = DPGP(X_train, Y_train, init_K=7, kernel=kernel, plot_conv=False,
            plot_sol=True)
dpgp.train()

print('\n ---------------------MODEL PARAMETERS DPGP-----------------------')
print(' Number of components identified, K = ', dpgp.K_opt)
print('Proportionalities: ', dpgp.pies)
print('Noise Stds: ', dpgp.stds, '\n')

print('DPGP hyperparameters \n', dpgp.kernel_, '\n')

# Identify relevant inputs
hyper = np.exp(dpgp.kernel_.theta)
ls = hyper[1:10]
relevant = np.where(ls < 10)[0]
inputs = X_df.columns

print('Relevant inputs:')
for i in range(len(relevant)):
    print(inputs[relevant[i]], str(ls[relevant[i]]))

# ----------------------------------------------------------------------------
# FUTURE PREDICTIONS
# ----------------------------------------------------------------------------

mu, std = dpgp.predict(X_train)

# Un-normalised data
mu = mu*Y_raw_std + Y_raw_mean
std = std*Y_raw_std
Y_raw = Y_train*Y_raw_std + Y_raw_mean
    
# ----------------------------------------------------------------------------
# STANDARD GP
# ----------------------------------------------------------------------------

gp = GPR(kernel=kernel, alpha=0, n_restarts_optimizer = 2,
         normalize_y = False).fit(X_train, Y_train)
muGP, stdGP = gp.predict(X_train, return_std=True)

# Un-normalised standard GP predictions
mu0 = muGP*Y_raw_std + Y_raw_mean
std0 = stdGP*Y_raw_std

# ----------------------------------------------------------------------------
# ROBUST GP (EMGP)
# ----------------------------------------------------------------------------
mixGP = EMGP(X_train, Y_train, init_K=3, kernel=kernel, normalise_y=True,
             N_iter=10, plot_conv=True)
mixGP.train()
mumix, std = mixGP.predict(np.vstack(X_train), return_std=True)

# Unormalised mean and std
muMix = mumix*mixGP.Y_std + mixGP.Y_mu
#-----------------------------------------------------------------------------
# temp PLOT (STANDARD GP CONFIDENCE BOUNDS)
#-----------------------------------------------------------------------------

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig.autofmt_xdate()
ax.fill_between(dt_train, mu0 + 3.2*std0, mu0 - 3.2*std0,
                alpha=0.5, color='orange', label='Confidence \nBounds (GP)')
ax.plot(dt_train, Y_raw, color="black", linewidth = 2.5, label="Scanner")
ax.plot(dt_train, mu0, color="yellow", linewidth = 2.5, label="GP expert")
ax.plot(dt_train, muMix, color="green", linewidth = 2.5, label="RGP(Stegle)-Local")
ax.plot(dt_train, mu, color="red", linewidth = 2.5, label="DPGP expert")
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

plt.show()

#-----------------------------------------------------------------------------
# DPGP REGRESSION CONFIDENCE BOUNDS
#-----------------------------------------------------------------------------

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig.autofmt_xdate()
ax.fill_between(dt_train, mu[:,0] + 3.2*std[:,0], mu[:,0] - 3.2*std[:,0],
                alpha=0.5, color='lightcoral',
                label='Confidence \nBounds (DPGP)')
ax.plot(dt_train, Y_raw, color="black", linewidth = 2.5, label="Scanner")
ax.plot(dt_train, mu0, color="yellow", linewidth = 2.5, label="GP expert")
ax.plot(dt_train, muMix, color="green", linewidth = 2.5, label="RGP(Stegle)-Local")
ax.plot(dt_train, mu, color="red", linewidth = 2.5, label="DPGP expert")
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

plt.show()

# ----------------------------------------------------------------------------
# PCA and PLOTS
# ----------------------------------------------------------------------------
pca = PCA(n_components=2)
pca.fit(X)
Xt = pca.transform(X)

# PCA on training data
Xt_train = pca.transform(X_train)
    
# Plot at each 1000 points
fig, ax = plt.subplots()
ax.plot(Xt[:, 0], Xt[:, 1], 'o', markersize=0.9, c='black',
        label='Available training data', alpha=0.6)
ax.plot(Xt_train[:, 0], Xt_train[:, 1], 'o', markersize=0.9, c='orange',
        label='Used Training data', alpha=0.6)
ax.set_xlim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 0])))
ax.set_ylim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 1])))

#-----------------------------------------------------------------------------
# CLUSTERING PLOT
#-----------------------------------------------------------------------------

color_iter = ['lightgreen', 'orange','red', 'brown','black']

# DP-GP
enumerate_K = [i for i in range(dpgp.K_opt)]

fig, ax = plt.subplots()
# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig.autofmt_xdate()
ax.set_title(" Clustering performance", fontsize=18)
if dpgp.K_opt != 1:
    for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
        ax.plot(dt_train[dpgp.indices[k]], Y_raw[dpgp.indices[k]],
                'o',color=c, markersize = 8, label='Noise Level '+str(k))
ax.plot(dt_train, mu, color="green", linewidth = 2, label=" DPGP")
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

plt.show()