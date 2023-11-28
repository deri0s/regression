import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('..')
sys.path.append('../..')
from sklearn.decomposition import PCA
import data_processing_methods as dpm

plt.close('all')

"""
PCA analysis
"""

# ----------------------------------------------------------------------------
# USER OPTIONS
# ----------------------------------------------------------------------------

# Choose the scanner to read data
scanner = 'MK4'

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
    file_name = ('../Input Post-Processing ' + str(i) + ' ' +
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
T_df = pd.read_excel('../Input Post-Processing 4 ISRA timelags.xlsx',
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

# ----------------------------------------------------------------------------
# TRAINING DATA
# ----------------------------------------------------------------------------

# Select training dataset
start = 0
end = 10000

# Save memory
del X_df, Y_df, Y_raw_df, dpm

# The training data
X_train = X[start: end]
Y_train = Y_raw_stand[start: end]
Y_filt = Y[start:end]
dt_train = date_time[start:end]
N_train = len(Y_train)
D = np.shape(X_train)[1]

# ----------------------------------------------------------------------------
# PCA and PLOTS
# ----------------------------------------------------------------------------
pca = PCA(n_components=2)
pca.fit(X_train)
Xt = pca.transform(X_train)

# Percentage of the variance explained by the PCs 1 and 2
print('Percentage: ', pca.explained_variance_ratio_,
      ' total: ', np.sum(pca.explained_variance_ratio_), '%')

print('\nThe most important features for PC1 are: \n',
      to_retain[np.argmax(pca.components_[0,:])], '\n',
      to_retain[np.argmin(pca.components_[0,:])])
print('\nThe most important features for PC2 are: \n',
      to_retain[np.argmax(pca.components_[1,:])], '\n',
      to_retain[np.argmin(pca.components_[1,:])])

# PCA on test data
Xt_test = []
Xt_testc = []
for i in range(4):
    print('inicio: ', end + (i*1000), ' end: ', end + (i +1)*1000)
    # At each 1000 points
    Xt_test.append(pca.transform(X[end + (i*1000): end + (i +1)*1000]))
    # Comulative 1000 points
    Xt_testc.append(pca.transform(X[end: end + (i +1)*1000]))

# ----------------------------------------------------------------------------
# PCA test data plots
# ----------------------------------------------------------------------------
# Plot at each 1000 points
fig, ax = plt.subplots(nrows=2, ncols=2)
ax = ax.flatten()
plt.suptitle('Testing data')
for i in range(np.shape(Xt_test)[0]):
    ax[i].plot(Xt[:, 0], Xt[:, 1], 'o', markersize=0.9, c='black',
            label='Training data with N = $10^4$', alpha=0.6)
    ax[i].plot(Xt_test[i][:, 0], Xt_test[i][:, 1], 'o', markersize=0.9,
            c='orange', label='4000 of testing data', alpha=0.6)
    ax[i].set_xlim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 0])))
    ax[i].set_ylim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 1])))

# Plot at each 1000 points
fig, ax = plt.subplots(nrows=2, ncols=2)
ax = ax.flatten()
plt.suptitle('Testing data')
for i in range(np.shape(Xt_test)[0]):
    ax[i].plot(Xt[:, 0], Xt[:, 1], 'o', markersize=0.9, c='black',
            label='Training data with N = $10^4$', alpha=0.6)
    ax[i].plot(Xt_testc[i][:, 0], Xt_testc[i][:, 1], 'o', markersize=0.9,
            c='orange', label='4000 of testing data', alpha=0.6)
    ax[i].set_xlim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 0])))
    ax[i].set_ylim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 1])))


# ----------------------------------------------------------------------------
# PCA training data plots
# ----------------------------------------------------------------------------
Xt_test = []
fig, ax = plt.subplots(nrows=2, ncols=2)
ax = ax.flatten()
plt.suptitle('Training data')
print('Where the training data overlaps the testing data \n')
for i in range(4):
    print('inicio: ', start + (i*3300), ' end: ', start + (i +1)*3300)
    Xt_test.append(pca.transform(X[start + (i*3300): start + (i +1)*3300]))
    ax[i].plot(Xt[:, 0], Xt[:, 1], 'o', markersize=0.9, c='black',
            label='Training data with N = $10^4$', alpha=0.6)
    ax[i].plot(Xt_test[i][:, 0], Xt_test[i][:, 1], 'o', markersize=0.9,
            c='orange', label='4000 of testing data', alpha=0.6)
    ax[i].set_xlim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 0])))
    ax[i].set_ylim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 1])))
    
# 3D PLOT
if pca.n_components > 2:
    fig, ax = plt.subplots()
    ax = fig.gca(projection='3d')
    
    zline = np.linspace(0, 15, 1000)
    xline = np.sin(zline)
    yline = np.cos(zline)
    ax.plot3D(Xt[:, 0], Xt[:, 1], Xt[:, 2], 'red')

plt.show()

# DateTime where the training and testing data overlapped
print('Training data from ', date_time[6600], ' to ', date_time[9900])
print('Testing data from ', date_time[13000], ' to ', date_time[14000])