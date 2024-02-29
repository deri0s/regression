import paths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as ss
from sklearn.preprocessing import MinMaxScaler
from NSG import data_processing_methods as dpm

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
# standardise y_raw
# scaler = ss()
scaler = MinMaxScaler(feature_range=(0,1))
y = scaler.fit_transform(np.vstack(y_raw))

# Extract corresponding time stamps. Note this essentially just
# removes the first max_lag points from the date_time array.
date_time = dpm.adjust_time_lag(y_df['Time stamp'].values,
                                shift=0,
                                to_remove=max_lag)

# Train and test data
val_split = 0.15
N, D = np.shape(X)
N_train = N

test_range = range(N)
X_test = X[test_range]
dt_test, y_test = date_time[test_range], y0[test_range]


"""
Neural Network
"""
import os
from tensorflow import keras
from sklearn.metrics import mean_absolute_error as mae

# Load trained model
path_file = os.getcwd()+'\\NSG\\regression\\Neural Networks'
name = '\\3HL_128_units_Nonstandardised_'
model1 = keras.models.load_model(path_file+name)

# Predictions on test data
yNN = model1.predict(X_test)


"""
Scalable Robust GP
"""
import pickle

file_name = 'ddpgp_N5901_scaler_ss.pkl'
file_path = os.getcwd()+'\\models\\trained_models\\'+file_name

with open(file_path,'rb') as f:
    ddpgp = pickle.load(f)

mu_stand, std_stand, betas = ddpgp.predict(X_test)

mu = scaler.inverse_transform(np.vstack(mu_stand))
std= scaler.inverse_transform(np.vstack(std_stand))

print('MAE')
print('NN: ', mae(y_test, yNN))
print('DDPGP: ', mae(y_test, mu))


"""
Plots
"""
# Region where test data is similar to the training data
similar = range(21000,21500)

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig.autofmt_xdate()
plt.fill_between(dt_test[N_train-int(N_train*val_split):], 50, color='pink', label='test data')
ax.plot(dt_test, y_raw, color="grey", linewidth = 2.5, label="Raw")
ax.plot(dt_test, y_test, color="blue", linewidth = 2.5, label="Conditioned")
ax.plot(dt_test, yNN, color="red", linewidth = 2.5, label="NN")
ax.plot(dt_test, mu, color="orange", linewidth = 2.5, label="DDPGP")
plt.fill_between(date_time[similar], 50, color='lightgreen', alpha=0.6,
                 label='test data similar to training')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":12}, facecolor="white", framealpha=1.0)

plt.show()