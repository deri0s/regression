import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as ss
from NSG import data_processing_methods as dpm
from NSG import *
import paths

"""
NSG data
"""
# NSG post processes data location
file = paths.get_data_path('NSG_data.xlsx')

# Training df
X_df = pd.read_excel(file, sheet_name='X_training_stand')
y_df = pd.read_excel(file, sheet_name='y_training')
y_raw_df = pd.read_excel(file, sheet_name='y_raw_training')
t_df = pd.read_excel(file, sheet_name='time')

# Pre-Process training data
X, y0, N0, D, max_lag, time_lags = dpm.align_arrays(X_df, y_df, t_df)
scaler = ss().fit(np.vstack(y0))
y = scaler.transform(np.vstack(y0))

# Process raw targets
# Just removes the first max_lag points from the date_time array.
y_raw = dpm.adjust_time_lag(y_raw_df['raw_furnace_faults'].values,
                            shift=0,
                            to_remove=max_lag)

# Check if the standardisation of y-raw is appropriate.
# Useful to see what data we use to train the DP-GP
scaler_raw = ss().fit(np.vstack(y_raw))
y_raw_stand = scaler_raw.transform(np.vstack(y_raw))

# Extract corresponding time stamps. Note this essentially just
# removes the first max_lag points from the date_time array.
date_time = dpm.adjust_time_lag(y_df['Time stamp'].values,
                                shift=0,
                                to_remove=max_lag)

print('N: ', np.shape(y_raw))

"""----------------------------------------------------------------------------
PCA Analysis
"""

from sklearn.decomposition import PCA
# Save memory
# del X_df, y_df

N, D = np.shape(X)

# Training data
N_train = 20000
X_train = X[0:N_train]
dt_train = date_time

# ----------------------------------------------------------------------------
# PCA and PLOTS
# ----------------------------------------------------------------------------
pca = PCA(n_components=2)
pca.fit(X_train)
Xt = pca.transform(X_train)

# Percentage of the variance explained by the PCs 1 and 2
print('Percentage: ', pca.explained_variance_ratio_,
      ' total: ', np.sum(pca.explained_variance_ratio_), '%')

# What are the inputs that provide the most information?
input_labels = ['10091 Furnace Load',
                '10271 C9 (T012) Upstream Refiner',
                '2922 Closed Bottom Temperature - Downstream Working End (PV)',
                '2921 Closed Bottom Temperature - Upstream Working End (PV)',
                '2918 Closed Bottom Temperature - Port 6 (PV)',
                '2923 Filling Pocket Closed Bottom Temperature Centre (PV)',
                '7546 Open Crown Temperature - Port 1 (PV)',
                '7746 Open Crown Temperature - Port 2 (PV)',
                '7522 Open Crown Temperature - Port 4 (PV)',
                '7483 Open Crown Temperature - Port 6 (PV)']

print('\nThe most important features for PC1 are: \n',
      input_labels[np.argmax(pca.components_[0,:])], '\n',
      input_labels[np.argmin(pca.components_[0,:])])
print('\nThe most important features for PC2 are: \n',
      input_labels[np.argmax(pca.components_[1,:])], '\n',
      input_labels[np.argmin(pca.components_[1,:])])

# PCA on test data
X_test = X[N_train:N]
Xt_test = []
Xt_testc = []
increment = 500
start = 0

for i in range(3):
    # At each `increment` points
    Xt_test.append(pca.transform(X_test[start:increment]))
    # Comulative
    Xt_testc.append(pca.transform(X_test[0:increment]))
    start += increment
    increment += increment

# ----------------------------------------------------------------------------
# PCA test data plots
# ----------------------------------------------------------------------------
# Plot at each `increment` points
fig, ax = plt.subplots(nrows=2, ncols=2)
ax = ax.flatten()
plt.suptitle('Testing data')
for i in range(np.shape(Xt_test)[0]):
    ax[i].plot(Xt[:, 0], Xt[:, 1], 'o', markersize=0.9, c='black',
               label='Training data with N = $N_train$', alpha=0.6)
    ax[i].plot(Xt_test[i][:, 0], Xt_test[i][:, 1], 'o', markersize=0.9,
               c='orange', label='$increment$ of testing data', alpha=0.6)
    ax[i].annotate(str(i), xy=(np.max(np.max(Xt[:, 0]))/2, np.max(np.max(Xt[:, 0]))/2),
                   c='red', fontsize=20)
    ax[i].set_xlim(np.min(Xt[:, 0])-1, np.max(np.max(Xt[:, 0])))
    ax[i].set_ylim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 1])))

# DateTime where the training and testing data overlapped
print('\n Test data that shows to be within a training region identified')
print(f"from {date_time[N_train+1000]} to {date_time[N_train+1500]}")

similar = range(N_train+1000,N_train+1500)

"""
-------------------------------------------------------------------------------
REGRESSION PLOTS
-------------------------------------------------------------------------------
"""
fig = plt.figure()
fig.autofmt_xdate()
plt.plot(date_time, y_raw, linewidth = 2.5, c='grey', label='Raw')
plt.plot(date_time, y_raw_stand, linewidth = 2.5, c='purple', label='Raw Standard(SS)')
plt.plot(date_time, y, linewidth = 2.5, c='red', label='Standardised(SS)')
plt.plot(date_time, y0, linewidth = 2.5, c='blue', label='Nonstandardised')
plt.fill_between(date_time[similar], 50, color='pink', label='test data similar to training')
plt.title('NSG Fault Density Data')
plt.legend()

"""
TARGET ANALYSIS

1) Use the statsmodels methods to extract the target's: trend, seasonal,
and data not explained by the previous two (residuals in statsmodels)
      - To do this I need to input to the statsmodels methods continous data

2) Use the pandas method `autocorrelation_plot` to determine if there is the
targets show autocorrelation
"""

from statsmodels.tsa.seasonal import MSTL
from pandas.plotting import autocorrelation_plot as autoplot

# Identify jumps occurring at the minute level
date_time = [pd.Timestamp(date_time[i]) for i in range(len(date_time))]

min_jumps = [i for i,v in enumerate(range(1, len(date_time))) if (date_time[i-1].minute - date_time[i].minute) == 20]
jumps = y_df['Time stamp'][min_jumps]
print('\njumps: \n', jumps)

dtc = date_time[jumps.index[1]+1: jumps.index[2]]
yc = y0[jumps.index[1]+1: jumps.index[2]]
y_rawc = y_raw[jumps.index[1]+1: jumps.index[2]]

# similar = range(N_train+1000,N_train+1500)
fig = plt.figure()
fig.autofmt_xdate()
plt.plot(date_time, y_raw, linewidth = 2.5, c='grey', label='Raw')
plt.plot(date_time, y0, linewidth = 2.5, c='blue', label='Conditioned')
for j in jumps.index:
    plt.axvline(date_time[j], linestyle='--', linewidth=3, color='lime', label='<- train | test ->'+str(j))
plt.title('Data jumps')
plt.legend()

# get the second continuous data region
fig = plt.figure()
fig.autofmt_xdate()
plt.plot(dtc, y_rawc, linewidth = 2.5, c='grey', label='Raw')
plt.plot(dtc, yc, linewidth = 2.5, c='blue', label='Conditioned')
plt.title('Continous region')
plt.legend()

min_per = 24*3
week_per = min_per*7
mstl = MSTL(yc, periods=[min_per, week_per])
res = mstl.fit()
res.plot()

plt.figure()
autoplot(yc)

plt.show()