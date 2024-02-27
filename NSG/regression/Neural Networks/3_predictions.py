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

""" Specify scaler here if the data was standardised """
# scaler = MinMaxScaler(feature_range=(0,1))
scaler = ss()
y = scaler.fit_transform(np.vstack(y0))

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
val_split = 0.15
N, D = np.shape(X)
N_train = N
X_train, y_train = X[0:N_train], y[0:N_train]

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
# name = '\\3HL_64_units_Standardised_relu_B5518_best_testing'
name = '\\4HL_128_units_B64_Nonstandardised'
model1 = keras.models.load_model(path_file+name)
model2 = keras.models.load_model(path_file+'\\3HL_128_units_Nonstandardised_')

# Predictions on test data
yNN1 = model1.predict(X_test)
yNN2 = model2.predict(X_test)

if 'Standardised' in name:
    print('standardised')
    yNN1 = scaler.inverse_transform(yNN1)
else:
    print('nonstandardised')

print('MAE')
print('Stand: ', mae(y_test, yNN1))
print('Non-stand: ', mae(y_test, yNN2))


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
ax.plot(dt_test, yNN1, color="red", linewidth = 2.5, label="NN-Standard")
ax.plot(dt_test, yNN2, color="orange", linewidth = 2.5, label="NN-Nonstand")
plt.fill_between(date_time[similar], 50, color='lightgreen', alpha=0.6,
                 label='test data similar to training')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":12}, facecolor="white", framealpha=1.0)

plt.show()