import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
sys.path.insert(0, 'C:\Diego\PhD\Code\phdCode')
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from DDPGP import DistributedDPGP as DDPGP
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

    # The fourth file is used to create the testing data-frame
    if i == 4:
        X_df_test = X_df_test._append(pd.read_excel(file_name,
                                     sheet_name='input_data'))
        Y_df_test = Y_df_test._append(pd.read_excel(file_name,
                                     sheet_name='output_data'))
        Y_raw_df_test = Y_raw_df_test._append(pd.read_excel(file_name,
                                             sheet_name='raw_output_data'))
        
# Extract time lags from final file (should be the same for all)
T_df = pd.read_excel('NSG/Input Post-Processing 4 ISRA timelags.xlsx',
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

# Y_raw_test standardisation
Y_raw_mean = np.mean(Y_raw_test)
Y_raw_std = np.std(Y_raw_test)
Y_raw_test_stand = dpm.standardise(Y_raw_test, Y_raw_mean, Y_raw_std)

# Extract corresponding time stamps. Note this essentially just
# removes the first max_lag points from the date_time array.
date_time = dpm.adjust_time_lag(Y_df['Time stamp'].values,
                                shift=0,
                                to_remove=max_lag)

date_time_test = dpm.adjust_time_lag(Y_df_test['Time stamp'].values,
                                     shift=0, to_remove=max_lag)

# ----------------------------------------------------------------------------
# DPGP
# ----------------------------------------------------------------------------

# Select training dataset (N_max = 2600)
Ngps = 2
start = 1000
end0 =  2500
step = (end0 - start)/Ngps # 676

# Save memory
del X_df, X_df_test, Y_df, Y_df_test, T_df, Y_raw_df, Y_raw_df_test, dpm

# The training data
X_train = X_test[start: end0]
Y_train = Y_raw_test_stand[start: end0]
D = np.shape(X_train)[1]

# Length scales
ls = [7, 64, 7, 7.60, 7, 7, 7, 123, 76, 78]

# Kernels
se = 1**2 * RBF(length_scale=ls, length_scale_bounds=(0.05, 200))
wn = WhiteKernel(noise_level=0.61**2, noise_level_bounds=(1e-5, 1))

kernel = se + wn
del se, wn, Y

# dpgp = DDPGP(X_train, Y_train, N_GPs = Ngps, init_K=4, kernel=kernel)
# dpgp.train(tol=15)

# Select testing data
end = -1
X_test = X_test[start: end]
date_time = date_time_test[start: end]
Y_raw_test = Y_raw_test[start: end]
print("xtest: ", np.shape(X_test))
"""
mu_dpgp, std_dpgp, betas = dpgp.predict(X_test)

# The estimated GP hyperparameters
print('\nEstimated hyper DRGP: ', dpgp.rgps[0].kernel_)

# Unormalise predictions
mu = mu_dpgp*Y_raw_std + Y_raw_mean
std = std_dpgp*np.std(Y_raw)

##############################################################################
# Plot beta
##############################################################################
c = ['red', 'orange', 'blue', 'black', 'green', 'cyan', 'darkred', 'pink',
     'gray', 'magenta','lightgreen', 'darkblue', 'yellow']

fig, ax = plt.subplots()
fig.autofmt_xdate()
for k in range(Ngps):
    ax.plot(date_time, betas[:,k], color=c[k], linewidth=2,
            label='Beta: '+str(k))
    print('step: ', int(k*step))
    plt.axvline(date_time[int(k*step)], linestyle='--', linewidth=2,
                color='black')
    
plt.axvline(date_time[int((k + 1)*step)], linestyle='--', linewidth=3,
            color='lime')
ax.set_title(scanner + ' scanner')
ax.set_xlabel('Date-time')
ax.set_ylabel('Predictive contribution')
# plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
plt.show()

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig.autofmt_xdate()
ax.fill_between(date_time, mu[:,0] + 3*std[:,0], mu[:,0] - 3*std[:,0],
                alpha=0.5, color='pink',
                label='Confidence \nBounds (DRGPs)')
ax.plot(date_time, Y_raw_test, color='black', label='Fault density')
ax.plot(date_time, mu, color="red", linewidth = 2.5, label="DRGPs")

# Plot the limits of each expert
for s in range(Ngps):
    plt.axvline(date_time[int(s*step)], linestyle='--', linewidth=2,
                color='blue')
plt.axvline(date_time[int((k + 1)*step)], linestyle='--', linewidth=3,
            color='lime')
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

# PCA on test data
Xt_test = pca.transform(X_test)
    
# Plot at each 1000 points
fig, ax = plt.subplots()
ax.plot(Xt[:, 0], Xt[:, 1], 'o', markersize=0.9, c='black',
        label='Available training data', alpha=0.6)
ax.plot(Xt_train[:, 0], Xt_train[:, 1], 'o', markersize=0.9, c='orange',
        label='Used Training data', alpha=0.6)
ax.plot(Xt_test[:,0], Xt_test[:,1], 'o', markersize=0.9,
        c='blue', label='4000 of testing data', alpha=0.6)
ax.set_xlim(np.min(Xt[:, 0]-1), np.max(np.max(Xt[:, 0])))
ax.set_ylim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 1])))
plt.show()

#-----------------------------------------------------------------------------
# SAVE RESULTS IN A SPREADSHEET
#-----------------------------------------------------------------------------
# Arrange the results in a single array
d = {}
d_pca_training = {}
d_pca_testing = {}
d['DateTime'] = date_time
d['y_raw'] = Y_raw_test
d['mu'] = mu[:,0]
d['std'] = std[:,0]
d_pca_training['Xt_train_pc1'] = Xt_train[:,0]
d_pca_training['Xt_train_pc2'] = Xt_train[:,1]
d_pca_testing['Xt_test_pc1'] = Xt_test[:,0]
d_pca_testing['Xt_test_pc2'] = Xt_test[:,1]

results_regression = pd.DataFrame(d)
results_pca_training = pd.DataFrame(d_pca_training)
results_pca_testing = pd.DataFrame(d_pca_testing)

# Define an Excel writer object and the target file
writer = pd.ExcelWriter('DDPGP_predictions.xlsx')

# Save to spreadsheet
results_regression.to_excel(writer, sheet_name='regression', index=False)
results_pca_training.to_excel(writer, sheet_name='pca_training', index=False)
results_pca_testing.to_excel(writer, sheet_name='pca_testing', index=False)
writer._save()
"""