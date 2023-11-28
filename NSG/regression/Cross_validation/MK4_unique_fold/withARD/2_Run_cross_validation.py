import sys
sys.path.append('../../../../..')
sys.path.append('../../../../../..')
import xlsxwriter
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as coef
import data_processing_methods as dpm

from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from DPGP import *

"""
Apply the 1-fold cross-validation analysis to indentify the global GP
hyperparameters. A N_sim number of simulations are run where a DP-GP is
trained with initial hyperparameters sampled from a Uniform distribution.
The negavite log-likelihood, Mean Square Error measure and Coeficient of
determination are saved in a different worsheet for each k-fold simulation
"""

# plt.close('all')

# Create an Excel file where we are going to save the simulation results
workbook = xlsxwriter.Workbook('Results_9_10.xlsx')

#-----------------------------------------------------------------------------
# INITIALISE WORKSHEETS FOR EACH K-FOLD SIMULATION
#-----------------------------------------------------------------------------
    
worksheet = workbook.add_worksheet('Unique fold')

# Write the titles of the parameters and the Neg-Log-likelihood
font_title = workbook.add_format({'bold': True, 'font_color': 'blue'})
worksheet.write(0, 0, 'Initial ls', font_title)
worksheet.write(0, 11, 'ls', font_title)
worksheet.write(0, 22, 'Neg Log-likleihood', font_title)
worksheet.write(0, 23, 'MSE', font_title)
worksheet.write(0, 24,'R2', font_title)

#-----------------------------------------------------------------------------
# GET THE TRAINING AND VALIDATION DATA FOR EACH K-FOLD
#-----------------------------------------------------------------------------
 
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

# This ust removes the first max_lag points from the date_time array.
Y_raw = dpm.adjust_time_lag(Y_raw_df['raw_furnace_faults'].values,
                            shift=0,
                            to_remove=max_lag)

# Y_raw standardisation
Y_raw_mean = np.mean(Y_raw_df['raw_furnace_faults'].values)
Y_raw_std = np.std(Y_raw_df['raw_furnace_faults'].values)
Y_raw_stand = dpm.standardise(Y_raw, Y_raw_mean, Y_raw_std)

# Extract corresponding time stamps. Note this essentially just
# removes the first max_lag points from the date_time array.
date_time = dpm.adjust_time_lag(Y_df['Time stamp'].values, shift=0,
                               to_remove=max_lag)

# Training indices
start = 9*1000
end = 10*1000

# Save memory
del X_df
del Y_df
del T_df
del Y_raw_df
del stand_df
del dpm

# The training data
X_train = X[start: end]
y_norm = Y_raw_stand[start: end]
Y_filt = Y[start:end]
dt_train = date_time[start:end]
N_train = len(y_norm)
D = np.shape(X_train)[1]

#-----------------------------------------------------------------------------
# LOAD THE TEST DATA
#-----------------------------------------------------------------------------
test_df = pd.read_excel('Test_data_for_9_10.xlsx')
 
# The validation dataset
start_test = start + 1000
end_test = end + 1000
X_test = X[start_test: end_test]
y_test = test_df['mu_10_11'].values
dt_test = test_df['DateTime'].values
N_test = len(y_test)
    
#-------------------------------------------------------------------------
# START THE HYPERPARAMETER SAMPLINGS
#-------------------------------------------------------------------------

N_sim = 200
for i in range(N_sim):
    print('\n Simulation number: ', i)
    col = 0
    
    # Sample the itial hyperparameters from a uniform distribution
    lower_ls = 0.5
    ls = np.random.uniform(lower_ls, 180, D)
    std = 0.6               # Fixed
    
    # The SE kernel plus noise variance
    se = 1**2 * RBF(length_scale=ls, length_scale_bounds=(lower_ls, 200))
    wn = WhiteKernel(noise_level=std**2,noise_level_bounds=(1e-5, 2))
    kernel = se + wn
    
    # Train the DP-GP model
    dpgp = DPGP(X_train, y_norm, init_K=4, kernel=kernel)
    dpgp.train()
    
    # Save results if the model has found a solution, ignore otherwise
    if dpgp.hyperparameters[-1] != 1e-5:
        # Predictions at validation points
        mu_norm = dpgp.predict(X_test)[0]
        
        # Unormalised predictions
        mu = mu_norm*Y_raw_std + Y_raw_mean
        error = abs(mse(y_test, mu))
        coef_det = coef(y_test, mu)
        
        # Arrange the results in a single array
        elements = (ls,
                    np.vstack(dpgp.hyperparameters[1:-1]),
                    [dpgp.log_marginal_likelihood_value_],
                    [error], [coef_det])
        
        # Write the results from the simulation in an Excel file
        for move_elements in range(len(elements)):
            for h in elements[move_elements]:
                worksheet.write(i+1, col, h)
                col += 1

workbook.close()

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig.autofmt_xdate()
# ax.fill_between(dt_test, mu[:,0] + 3*std[:,0], mu[:,0] - 3*std[:,0],
#                 alpha=0.5, color='pink',
#                 label='Confidence \nBounds (DP-GP)')
ax.plot(dt_test, y_test, color='black', label='Fault density')
ax.plot(dt_test, mu, color="red", linewidth = 2.5, label="DP-GP")
plt.axvline(dt_train[-1], linestyle='--', linewidth=2, color='black')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
