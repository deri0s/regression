import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
sys.path.append('../../..')
sys.path.append('../../../..')
import data_processing_methods as dpm
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from DPGP import DirichletProcessGaussianProcess as DPGP

import xlsxwriter
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as coef


"""
Apply the 1-fold cross-validation analysis to indentify the global GP
hyperparameters. A N_sim number of simulations are run where a DP-GP is
trained with initial hyperparameters sampled from a Uniform distribution.
The negavite log-likelihood, Mean Square Error measure and Coeficient of
determination are saved in a different worsheet for each k-fold simulation
"""

plt.close('all')

# ----------------------------------------------------------------------------
# USER OPTIONS
# ----------------------------------------------------------------------------

# Create an Excel file where we are going to save the simulation results
expert = 8
workbook =xlsxwriter.Workbook('Results_'+str(expert)+'_'+str(expert+1)+'.xlsx')

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
T_df = pd.read_excel('../../../Input Post-Processing 4 ISRA timelags.xlsx',
                     sheet_name='time_lags')

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

#-----------------------------------------------------------------------------
# LOAD THE TEST DATA
#-----------------------------------------------------------------------------
test_df = pd.read_excel('Test_data_for_'+str(expert)+'_'+str(expert+1)+'.xlsx')
y_test = test_df['mu_'+str(expert+1)+'_'+str(expert+2)].values
step = len(y_test)

# ----------------------------------------------------------------------------
# TRAINING AND TESTING DATA SELECTION
# ----------------------------------------------------------------------------

# Training indices
start = expert*step
end = (expert + 1)*step

# Save memory
del X_df, Y_df, T_df, Y_raw_df, dpm

# The training data
X_train = X[start: end]
y_norm = Y_raw_stand[start: end]
Y_filt = Y[start:end]
dt_train = date_time[start:end]
N_train = len(y_norm)
D = np.shape(X_train)[1]
 
# The validation dataset
start_test = start + step
end_test = end + step
X_test = X[start_test: end_test]
dt_test = test_df['DateTime'].values
N_test = len(y_test)

#-----------------------------------------------------------------------------
# INITIALISE WORKSHEETS FOR EACH K-FOLD SIMULATION
#-----------------------------------------------------------------------------
    
worksheet = workbook.add_worksheet('Unique fold')

# Write the titles of the parameters and the Neg-Log-likelihood
font_title = workbook.add_format({'bold': True, 'font_color': 'blue'})
worksheet.write(0, 0, 'Initial ls', font_title)
worksheet.write(0, D, 'ls', font_title)
worksheet.write(0, 2*D, 'Neg Log-likleihood', font_title)
worksheet.write(0, 2*D + 1, 'MSE', font_title)
worksheet.write(0, 2*D + 2,'R2', font_title)

#-------------------------------------------------------------------------
# START THE HYPERPARAMETER SAMPLINGS
#-------------------------------------------------------------------------

N_sim = 200
for i in range(N_sim):
    print('\n Simulation number: ', i)
    col = 0
    
    # Sample the itial hyperparameters from a uniform distribution
    lower_ls = 0.3
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
