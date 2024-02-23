import paths
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler as ss
from sklearn.preprocessing import MinMaxScaler
from BayesianNonparametrics.DPGP import DirichletProcessGaussianProcess as DPGP
from BayesianNonparametrics.DDPGP import DistributedDPGP as DDPGP
from NSG import data_processing_methods as dpm
from sklearn.decomposition import PCA

"""
NSG data
"""
# NSG post processes data location
file = paths.get_data_path('NSG_data.xlsx')
stand = 1
scaler_type = 'minmax'

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
N_train = 550
N = 600
stand = 1

# Scale data using MinMaxScaler. Do not use StandardScaler since the NN training stage
# struggles to find a solution.
if scaler_type == 'minmax':
    scaler = MinMaxScaler(feature_range=(0,1))
elif scaler_type == 'ss':
    scaler = ss()
else:
    assert False, f"scaler {stand} not a valid"

def get_train_test(X: np.array, y_orig: np.array, stand: bool, scaler=None):
    if stand:
        y_s = scaler.fit_transform(np.vstack(y_orig))
        X_train, y_train = X[0:N_train], y_s[0:N_train]
        X_test, y_test = X[0:N], y_orig[0:N]
        label = 'Standardised'
    else:
        X_train, y_train = X[0:N_train], y_orig[0:N_train]
        X_test, y_test = X[0:N], y_orig[0:N]
        label = 'Nonstandardised'

    return X_train, y_train, X_test, y_test, label

X_train, y_train, X_test, y_test, stand_label = get_train_test(X, y_raw, stand, scaler=scaler)
date_time = date_time[0:N]
y_rect = y0[0:N]

"""
DPGP regression
"""
# Save memory
del X_df, y_df, dpm

# Length scales
ls = [0.0612, 3.72, 200, 200, 200, 200, 4.35, 0.691, 200, 200]

# Kernels
se = 1**2 * RBF(length_scale=ls, length_scale_bounds=(0.05, 200))
wn = WhiteKernel(noise_level=0.61**2, noise_level_bounds=(1e-5, 1))

kernel = se + wn

dpgp = DPGP(X_train, y_train, init_K=7, kernel=kernel, plot_conv=True, plot_sol=True)
dpgp.train()
# predictions
mu_dpgp, std_dpgp = dpgp.predict(X_test)

"""
DDPGP regression
"""
N_gps = 2
dgp = DDPGP(X_train, y_train, N_GPs=N_gps, init_K=7, kernel=kernel)
dgp.train()
# predictions
mu_, std_, betas = dgp.predict(X_test)

# The estimated GP hyperparameters
print('\nEstimated hyper DRGP: ', dgp.rgps[0].kernel_)

# Unormalise predictions
if stand_label:
     mu = scaler.inverse_transform(mu_dpgp)
     std= scaler.inverse_transform(std_dpgp)
     muMix = scaler.inverse_transform(mu_)
     stdMix= scaler.inverse_transform(std_)

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()

# ax.fill_between(date_time,
#                 mu[:,0] + 3*std[:,0], mu[:,0] - 3*std[:,0],
#                 alpha=0.5, color='pink',
#                 label='Confidence \nBounds (DPGP)')
ax.plot(date_time, y_raw[0:N], color='grey', label='Raw')
ax.plot(date_time, y_raw[0:N], color='blue', label='Filtered')
ax.plot(date_time, mu, color="red", linewidth = 2.5, label="DPGP")
ax.plot(date_time, muMix, color="limegreen", linewidth = 2.5, label="DDPGP")
plt.axvline(date_time[N_train], linestyle='--', linewidth=3,
            color='black')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

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
        label='Available training data', alpha=0.9)
ax.plot(Xt_train[:, 0], Xt_train[:, 1], 'o', markersize=5.9, c='orange',
        label='Used Training data', alpha=0.6)
ax.plot(Xt_test[:,0], Xt_test[:,1], '*', markersize=5.5,
        c='pink', label='test data', alpha=0.6)
ax.set_xlim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 0])))
ax.set_ylim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 1])))
plt.show()