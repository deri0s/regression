import sys
sys.path.insert(0, 'C:\Diego\PhD\Code\phdCode')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as ss
from NSG import data_processing_methods as dpm

"""
NSG data
"""
# NSG post processes data location
file = 'NSG_data.xlsx'

# Training df
X_df = pd.read_excel(file, sheet_name='X_training')
y_df = pd.read_excel(file, sheet_name='y_training')
y_raw_df = pd.read_excel(file, sheet_name='y_raw_training')
t_df = pd.read_excel(file, sheet_name='time')

# Pre-Process training data
X0, y0, N0, D, max_lag, time_lags = dpm.align_arrays(X_df, y_df, t_df)
scaler = ss().fit(np.vstack(y0))
X = ss().fit(X0).transform(X0)
y = scaler.transform(np.vstack(y0))

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
N_train = 18000
X_train, y_train = X[0:N_train], y[0:N_train]

test_range = range(N)
X_test = X[test_range]
dt_test, y_test = date_time[test_range], y0[test_range]


"""
Neural Network
"""
import os
from tensorflow import keras

# Load trained model
location = os.getcwd()+'\\regression\\Neural Networks\\Varying Hyperparameters\\Batch'
model = keras.models.load_model(location+'\\1HL4__relu_B512')
model.summary()

# Predictions on test data
yp_stand = model.predict(X_test)
yNN = scaler.inverse_transform(yp_stand)


"""
Plots
"""
fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig.autofmt_xdate()
plt.axvline(date_time[N_train], linestyle='--', linewidth=3, color='lime', label='<- train | test ->')
# ax.plot(dt_train, y_raw, color="grey", linewidth = 2.5, label="Raw")
ax.plot(dt_test, y_test, color="blue", linewidth = 2.5, label="Conditioned")
ax.plot(dt_test, yNN, '--', color="red", linewidth = 2.5, label="NN")
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":12}, facecolor="white", framealpha=1.0)

plt.show()