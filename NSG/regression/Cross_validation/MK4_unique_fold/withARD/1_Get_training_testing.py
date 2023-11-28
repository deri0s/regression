import sys
sys.path.append('../../..')
sys.path.append('../../../..')
sys.path.append('../../../../../..')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import data_processing_methods as dpm

from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from DPGP import *

"""
This only generates the predictive mean of the M+1 expert. This data is the
test data for the M expert in the cross-validation analysis
"""

# plt.close('all')
# ----------------------------------------------------------------------------
# USER OPTIONS
# ----------------------------------------------------------------------------

to_retain = ['10271 Open Crown Temperature - Upstream Refiner (PV)',
             '2921 Closed Bottom Temperature - Upstream Working End (PV)',
             '2922 Closed Bottom Temperature - Downstream Working End (PV)',
             '2913 Closed Bottom Temperature - Port 1 (PV)',
             '2918 Closed Bottom Temperature - Port 6 (PV)',
             '7546 Open Crown Temperature - Port 1 (PV)',
             '7746 Open Crown Temperature - Port 2 (PV)',
             '7483 Open Crown Temperature - Port 6 (PV)',
             '10091 Furnace Load',
             '1650 Combustion Air Temperature Measurement',
             '10425 Calculated Cullet Ratio']

# ----------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------

# Load data frames
scanner = 'MK4'

if scanner == 'MK4':
    file_name = '../../../../../../Input Post-Processing 3 MK4 2021_07_01.xlsx'
else:
    file_name = '../../../../../../Input Post-Processing 3 ISRA 2021_07_05.xlsx'
    
X_df = pd.read_excel(file_name, sheet_name='input_data')
Y_df = pd.read_excel(file_name, sheet_name='output_data')
T_df = pd.read_excel(file_name, sheet_name='time_lags')
Y_raw_df = pd.read_excel(file_name, sheet_name='raw_output_data')
stand_df = pd.read_excel(file_name, sheet_name='Statistics')

# ----------------------------------------------------------------------------
# MANUALLY IGNORE INPUTS
# ----------------------------------------------------------------------------

input_names = X_df.columns
for name in input_names:
    if name not in to_retain:
        X_df.drop(columns=name, inplace=True)
        T_df.drop(columns=name, inplace=True)

# ----------------------------------------------------------------------------
# PRE-PROCESSING
# ----------------------------------------------------------------------------
# Note that, for now, standardisation is performed using statistics from
# all of the data (rather than just the training data). This is something
# we may want to consider changing later.

# Finding raw fault density mean and std (at training points)
Y_mean = np.mean(Y_df['furnace_faults'].values)
Y_std = np.std(Y_df['furnace_faults'].values)

# Standardise training data
for i in range(np.shape(X_df)[1]):
    tag_name = X_df.columns[i]

    # Re-write X_df now with standardise data (at training points)
    X_df[tag_name] = dpm.standardise(X_df.iloc[:, i],
                                     np.mean(X_df.iloc[:, i]),
                                     np.std(X_df.iloc[:, i]))

Y_df['furnace_faults'] = dpm.standardise(Y_df['furnace_faults'].values,
                                         Y_mean,
                                         Y_std)

# Process data
X, Y, N, D, max_lag, timelags  = dpm.align_arrays(X_df, Y_df, T_df)

# Note this essentially just removes the first max_lag
# points from the date_time array.
Y_raw = dpm.adjust_time_lag(Y_raw_df['raw_furnace_faults'].values, shift=0,
                            to_remove=max_lag)

# Y_raw standardisation
Y_raw_mean = np.mean(Y_raw_df['raw_furnace_faults'].values)
Y_raw_std = np.std(Y_raw_df['raw_furnace_faults'].values)
Y_raw_stand = dpm.standardise(Y_raw, Y_raw_mean, Y_raw_std)

# Extract corresponding time stamps. Note this essentially just
# removes the first max_lag points from the date_time array.
date_time = dpm.adjust_time_lag(Y_df['Time stamp'].values, shift=0,
                               to_remove=max_lag)

# The training dataset that is going to be used as the test data for the M expert
M = 10
M_plus = M + 1
start = M*1000
end = M_plus*1000

# Save memory
del X_df
del Y_df
del T_df
del Y_raw_df
del stand_df
del dpm

# The training data
X_train = X[start: end]
Y_train = Y_raw_stand[start: end]
Y_filt = Y[start:end]
dt_train = date_time[start:end]
N_train = len(Y_train)

start_test = start
end_test = end # are 6 days
X_test = X[start_test:end_test]
dt_test = date_time[start_test:end_test]
N_test = len(dt_test[start_test:end_test])

# ----------------------------------------------------------------------------
# GP REGRESSION
# ----------------------------------------------------------------------------
ls = [2.84, 64, 200, 2.7, 2.79, 3.95, 0.564, 167, 200, 1.07, 64]

se = 1**2 * RBF(length_scale=ls, length_scale_bounds=(0.1, 2))

wn = WhiteKernel(noise_level=0.61**2, noise_level_bounds=(1e-5, 1))

kernel = se + wn

del se
del wn

# DPGP
dpgp = DPGP(X_train, Y_train, init_K=4, kernel=kernel, plot_conv=False)
dpgp.train()
mu, std = dpgp.predict(X_test)

print('\n MODEL PARAMETERS DP-GP: \n')
print(' Number of components identified, K = ', dpgp.K_opt)
print('Proportionalities: ', dpgp.pies)
print('Noise Stds: ', dpgp.stds)
print('\nHyperparameters: ', dpgp.kernel)

# Un-normalised data
mu = mu*Y_raw_std + Y_raw_mean
std = std*np.std(Y_raw[dpgp.indices[0]])
Y_raw = Y_raw_stand*Y_raw_std + Y_raw_mean
Y_raw = Y_raw[start_test:end_test]

d = {}
d['DateTime'] = dt_test
d['mu_'+str(M)+'_'+str(M_plus)] = mu[:,0]
test_data_df = pd.DataFrame(d)

M_string = str(M-1)+'_'+str(M_plus-1)
# Define an Excel writer object
writer = pd.ExcelWriter('Test_data_for_' + M_string + '.xlsx')

# Save to spreadsheet
test_data_df.to_excel(writer, sheet_name='Test data', index='DateTime')
writer.save()

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig.autofmt_xdate()
ax.fill_between(dt_test, mu[:,0] + 3*std[:,0], mu[:,0] - 3*std[:,0],
                alpha=0.5, color='pink',
                label='Confidence \nBounds (DP-GP)')
ax.plot(dt_test, Y_raw, color='black', label='Fault density')
ax.plot(dt_test, mu, color="red", linewidth = 2.5, label="DP-GP")
plt.axvline(dt_train[-1], linestyle='--', linewidth=2, color='black')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

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
ax.plot(dt_test, mu, color="green", linewidth = 2, label=" DP-GP")
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

plt.show()