import paths
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import MinMaxScaler
from BayesianNonparametrics.DDPGP import DistributedDPGP as DDPGP
from NSG import data_processing_methods as dpm
from sklearn.decomposition import PCA

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

# Process raw targets
# Just removes the first max_lag points from the date_time array.
y_raw = dpm.adjust_time_lag(y_raw_df['raw_furnace_faults'].values,
                            shift=0,
                            to_remove=max_lag)

# Extract corresponding time stamps. Note this essentially just
# removes the first max_lag points from the date_time array.
date_time = dpm.adjust_time_lag(y_df['Time stamp'].values,
                                shift=0,
                                to_remove=max_lag)

# Train and test data
N, D = np.shape(X)
N_train = N
stand = 1

# Scale data using MinMaxScaler. Do not use StandardScaler since the NN training stage
# struggles to find a solution.
scaler = MinMaxScaler(feature_range=(0,1))

def get_train_test(X: np.array, y_orig: np.array, stand: bool):
    y = scaler.fit_transform(np.vstack(y0))

    if stand:
        X_train, y_train = X[0:N_train], y[0:N_train]
        X_test, y_test = X[0:N], y_orig[0:N]
        label = 'Standardised'
    else:
        X_train, y_train = X[0:N_train], y_orig[0:N_train]
        X_test, y_test = X[0:N], y_orig[0:N]
        label = 'Nonstandardised'

    return X_train, y_train, X_test, y_test, label

X_train, y_train, X_test, y_test, stand_label = get_train_test(X, y0, stand)
date_time = date_time[0:N]

"""
DPGP regression
"""
# Save memory
del X_df, y_df, dpm

# Length scales
ls = [7, 64, 7, 7.60, 7, 7, 7, 123, 76, 78]

# Kernels
se = 1**2 * RBF(length_scale=ls, length_scale_bounds=(0.05, 200))
wn = WhiteKernel(noise_level=0.61**2, noise_level_bounds=(1e-5, 1))

kernel = se + wn

N_gps = 40
dpgp = DDPGP(X_train, y_train, N_GPs=N_gps, init_K=4, kernel=kernel)
dpgp.train()

# predictions
mu_dpgp, std_dpgp, betas = dpgp.predict(X_test)

# The estimated GP hyperparameters
print('\nEstimated hyper DRGP: ', dpgp.rgps[0].kernel_)

# Unormalise predictions
if stand_label:
     mu = scaler.inverse_transform(mu_dpgp)

##############################################################################
# Plot beta
##############################################################################
c = ['red', 'orange', 'blue', 'black', 'green', 'cyan', 'darkred', 'pink',
     'gray', 'magenta','lightgreen', 'darkblue', 'yellow']

step = int(len(X_train)/N_gps)
fig, ax = plt.subplots()
fig.autofmt_xdate()
for k in range(N_gps):
    ax.plot(date_time, betas[:,k], color=c[k], linewidth=2,
            label='Beta: '+str(k))
    print('step: ', int(k*step))
    plt.axvline(date_time[int(k*step)], linestyle='--', linewidth=2,
                color='black')
    
plt.axvline(date_time[int((k + 1)*step)], linestyle='--', linewidth=3,
            color='lime')
ax.set_title('Que?')
ax.set_xlabel('Date-time')
ax.set_ylabel('Predictive contribution')
# plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig.autofmt_xdate()
# ax.fill_between(date_time, mu[:,0] + 3*std[:,0], mu[:,0] - 3*std[:,0],
#                 alpha=0.5, color='pink',
#                 label='Confidence \nBounds (DRGPs)')
ax.plot(date_time, y_raw, color='grey', label='Fault density')
ax.plot(date_time, mu, color="red", linewidth = 2.5, label="DRGPs")

# Plot the limits of each expert
for s in range(N_gps):
    plt.axvline(date_time[int(s*step)], linestyle='--', linewidth=2,
                color='blue')
plt.axvline(date_time[int((k + 1)*step)], linestyle='--', linewidth=3,
            color='lime')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

# ----------------------------------------------------------------------------
# PCA and PLOTS
# ----------------------------------------------------------------------------
# pca = PCA(n_components=2)
# pca.fit(X)
# Xt = pca.transform(X)

# # PCA on training data
# Xt_train = pca.transform(X_train)

# # PCA on test data
# Xt_test = pca.transform(X_test)
    
# # Plot at each 1000 points
# fig, ax = plt.subplots()
# ax.plot(Xt[:, 0], Xt[:, 1], 'o', markersize=0.9, c='black',
#         label='Available training data', alpha=0.6)
# ax.plot(Xt_train[:, 0], Xt_train[:, 1], 'o', markersize=0.9, c='orange',
#         label='Used Training data', alpha=0.6)
# ax.plot(Xt_test[:,0], Xt_test[:,1], 'o', markersize=0.9,
#         c='blue', label='4000 of testing data', alpha=0.6)
# ax.set_xlim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 0])))
# ax.set_ylim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 1])))
