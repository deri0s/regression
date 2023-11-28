import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
sys.path.append('../../..')
sys.path.append('../../../..')
import data_processing_methods as dpm
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from DPGP import DirichletProcessGaussianProcess as DPGP

"""
This only generates the predictive mean of the M+1 expert. This data is the
test data for the M expert in the cross-validation analysis
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
    file_name = ('../../../Input Post-Processing ' + str(i) + ' ' +
                 scanner + '.xlsx')

    # The first 3 files are appended to become a single training data-frame
    if i < 4:
        X_df = X_df.append(pd.read_excel(file_name,
                                         sheet_name='input_data'))
        Y_df = Y_df.append(pd.read_excel(file_name,
                                         sheet_name='output_data'))
        Y_raw_df = Y_raw_df.append(pd.read_excel(file_name,
                                   sheet_name='raw_output_data'))

    # The fourth file is used to create the testing data-frame
    if i == 4:
        X_df_test = X_df_test.append(pd.read_excel(file_name,
                                     sheet_name='input_data'))
        Y_df_test = Y_df_test.append(pd.read_excel(file_name,
                                     sheet_name='output_data'))
        Y_raw_df_test = Y_raw_df_test.append(pd.read_excel(file_name,
                                             sheet_name='raw_output_data'))

        # Extract time lags from final file (should be the same for all)
        # T_df = pd.read_excel(file_name, sheet_name='time_lags')

# Extract time lags from final file (should be the same for all)
T_df = pd.read_excel('../../../Input Post-Processing 4 ISRA timelags.xlsx',
                     sheet_name='time_lags')

print('X_df columns: ', np.shape(X_df))
# Check data frames are the correct size and have the same column names
assert np.all(X_df.columns == X_df_test.columns)
assert np.all(X_df.columns == T_df.columns)
assert len(X_df) == len(Y_df)
assert len(Y_df) == len(Y_raw_df)
assert len(X_df_test) == len(Y_df_test)
assert len(Y_df_test) == len(Y_raw_df_test)

# ----------------------------------------------------------------------------
# REMOVE INPUTS WE ARE NOT GOING TO USE
# ----------------------------------------------------------------------------

input_names = X_df.columns
for name in input_names:
    if name not in to_retain:
        X_df.drop(columns=name, inplace=True)
        X_df_test.drop(columns=name, inplace=True)
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
    # Re-write X_df_test now with standardise data (at training points)
    X_df_test[tag_name] = dpm.standardise(X_df_test.iloc[:, i],
                                          X_mean,
                                          X_std)

# Standardise testing data
Y_df['furnace_faults'] = dpm.standardise(Y_df['furnace_faults'].values,
                                         Y_mean,
                                         Y_std)

Y_df_test['furnace_faults'] = dpm.standardise(Y_df_test['furnace_faults'].values,
                                              Y_mean,
                                              Y_std)
# Process training data
X, Y, N, D, max_lag, time_lags = dpm.align_arrays(X_df,
                                                  Y_df,
                                                  T_df)

# Process testing data
X_test, Y_test, N_test, D, max_lag, time_lags = dpm.align_arrays(X_df_test,
                                                                 Y_df_test,
                                                                 T_df)

# Process raw target data in the same way as the post-processed
# target data. Note this essentially just removes the first max_lag
# points from the date_time array.
Y_raw = dpm.adjust_time_lag(Y_raw_df['raw_furnace_faults'].values,
                            shift=0,
                            to_remove=max_lag)

# This just removes the first max_lag points from the date_time array.
Y_raw = dpm.adjust_time_lag(Y_raw_df['raw_furnace_faults'].values, shift=0,
                            to_remove=max_lag)

# Y_raw standardisation
Y_raw_mean = np.mean(Y_raw_df['raw_furnace_faults'].values)
Y_raw_std = np.std(Y_raw_df['raw_furnace_faults'].values)
Y_raw_stand = dpm.standardise(Y_raw, Y_raw_mean, Y_raw_std)

# Y_raw_test does not need to be standardise
Y_raw_test = dpm.adjust_time_lag(Y_raw_df_test['raw_furnace_faults'].values,
                                 shift=0,
                                 to_remove=max_lag)

# Extract corresponding time stamps. Note this essentially just
# removes the first max_lag points from the date_time array.
date_time = dpm.adjust_time_lag(Y_df['Time stamp'].values,
                                shift=0,
                                to_remove=max_lag)

date_time_test = dpm.adjust_time_lag(Y_df_test['Time stamp'].values,
                                     shift=0, to_remove=max_lag)

# The training dataset that is going to be used as the test data for the
# M expert in the corss-validation analysis
M = 9
M_plus = M + 1
start = M*1000
end = M_plus*1001

# Save memory
del X_df, Y_df, T_df, Y_raw_df, dpm

# The training data
X_train = X[start: end]
Y_train = Y_raw_stand[start: end]
Y_filt = Y[start:end]
dt_train = date_time[start:end]

start_test = start
end_test = end # are 6 days
X_test = X[start_test:end_test]
dt_test = date_time[start_test:end_test]
N_test = len(dt_test[start_test:end_test])

# ----------------------------------------------------------------------------
# DPGP REGRESSION
# ----------------------------------------------------------------------------
# ls = [2.84, 64, 2.7, 2.79, 3.95, 0.564, 167, 200, 1.07, 64]
ls = [64, 64, 84, 20, 7.60, 20, 64, 123, 78, 78]

se = 1**2 * RBF(length_scale=ls, length_scale_bounds=(0.1, 200))
wn = WhiteKernel(noise_level=0.61**2, noise_level_bounds=(1e-5, 1))

kernel = se + wn

del se, Y, wn

# DPGP
dpgp = DPGP(X_train, Y_train, init_K=4, kernel=kernel, plot_conv=True)
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