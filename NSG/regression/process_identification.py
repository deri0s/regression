import paths
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler as ss
from sklearn.preprocessing import MinMaxScaler
from BayesianNonparametrics.DPGP import DirichletProcessGaussianProcess as DPGP
from NSG import data_processing_methods as dpm

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

N, D = np.shape(X)

# Scale data using MinMaxScaler. Do not use StandardScaler 
scaler = MinMaxScaler(feature_range=(0,1))
y_s = scaler.fit_transform(np.vstack(y_raw))

# define N elements
N_elemets = 22
h = N/N_elemets
print('N: ', N, ' N elements: ', 22, ' N-size: ', h)

# # Length scales
ls = [7, 64, 7, 7.60, 7, 7, 7, 123, 76, 78]

# Kernels
se = 1**2 * RBF(length_scale=ls, length_scale_bounds=(0.05, 200))
wn = WhiteKernel(noise_level=0.61**2, noise_level_bounds=(1e-5, 1))

kernel = se + wn

step = 0
dtl = []
mul = []
for i in range(N_elemets):
    start = int(step)
    end = int(step + h)
    print('i: ', i, 'start: ', start, ' end: ', end)

    # assemble training and test data
    X_train, y_train = X[start:end], y_s[start:end]
    X_test, y_test = X[start:end], y_raw[start:end]
    dt = date_time[start:end]

    # train model
    dpgp = DPGP(X_train, y_train, init_K=7, kernel=kernel, plot_conv=True)
    dpgp.train()

    # predictions
    mu_dpgp, std_dpgp = dpgp.predict(X_test)

    # Unormalise predictions
    mu = scaler.inverse_transform(mu_dpgp)
    std= scaler.inverse_transform(std_dpgp)

    dtl.append(dt)
    mul.append(mu)
    step += h

d = {'date_time': dtl, 'furnace_faults': mul}

df = pd.DataFrame(d)
df.to_csv('melting_process.csv', index=False)