import paths
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler as ss
from sklearn.preprocessing import MinMaxScaler
from BayesianNonparametrics.DPGP import DirichletProcessGaussianProcess as DPGP
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

# Scale data using MinMaxScaler. Do not use StandardScaler 
scaler = MinMaxScaler(feature_range=(0,1))

y_s = scaler.fit_transform(np.vstack(y_raw))

# Train and test data
N, D = np.shape(X)
start_train = 18
end_train = 22044
end_test = 22044
N_train = abs(end_train - start_train)

date_time = date_time[start_train:end_test]
y_raw = y_raw[start_train:end_test]
y_rect = y0[start_train:end_test]

print('date-time original: ', date_time[0])

"""
NSG data
"""
# NSG post processes data location
file = paths.get_data_path('melting_process.csv')
df = pd.read_csv(file)
# print(df.sort_values('date_time'))
# date_time = df['date_time'].values
mu = df['furnace_faults'].values

print('date-time DPGP: ', date_time[0])

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()

ax.set_xlabel(" Date-time", fontsize=14)
ax.plot(y_raw, color='grey', label='Raw')
# ax.plot(y_raw, 'o', color='grey', label='Raw')
# ax.plot(date_time, y_rect, color='blue', label='Filtered')
ax.plot(mu, color="red", linewidth = 2.5, label="DPGP")
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
plt.show()