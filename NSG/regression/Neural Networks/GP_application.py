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

start = 1000
end = 6000
dif = 6000
X_train, y_train = X[start:end], y[start:end]
X_test = X[end-dif:end+dif+8000]
# y_test doesn't need to be standardise
y_test = y0[end-dif:end+dif+8000]
y_raw = y_raw[end-dif:end+dif+8000]
dt_train = date_time[end-dif:end+dif+8000]


"""
Gaussian Process
"""
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Length scales
ls = [7, 64, 7, 7.60, 7, 7, 7, 123, 76, 78]

# Kernels
se = 1**2 * RBF(length_scale=ls, length_scale_bounds=(0.05, 200))
wn = WhiteKernel(noise_level=0.61**2, noise_level_bounds=(1e-5, 1))

kernel = se + wn
del se, wn

# Train and prediction
gp = GPR(kernel=kernel, alpha=0, n_restarts_optimizer = 2,
         normalize_y = False).fit(X_train, y_train)
mus, stds = gp.predict(X_test, return_std=True)

# Un-standardise targets and predictions
mu = scaler.inverse_transform(np.vstack(mus))


"""
Neural Network
"""
from tensorflow import keras
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# # Architecture
# model = keras.models.Sequential()
# model.add(Dense(128, name='hidden1', activation="relu", input_dim=D))
# model.add(Dense(32, name='hidden2', activation='relu'))
# model.add(Dense(8, name='hidden3', activation='relu'))
# model.add(Dense(1, name='output', activation='linear'))

# # Compilation
# model.compile(loss='mean_squared_error',
#               optimizer=keras.optimizers.Adam(lr=1e-3))

# # Patient early stopping
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

# # Fit the model
# iter = 1000
# B = 300
# val_split = 0.2
# history = model.fit(X_train, y_train, validation_split=val_split,
#                     epochs=iter, batch_size=B, verbose=2, callbacks=[es])

# # Plot accuracy of the model after each epoch.
# plt.figure()
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.title("B="+str(B))
# plt.xlabel("N of Epoch")
# plt.ylabel("Error (MAE)")
# plt.legend()

# Load trained model
model = keras.models.load_model('2HL_32relu_8relu_B3000_epoch10000')
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
plt.axvline(dt_train[start], linestyle='--', linewidth=3, color='lime', label='train->')
plt.axvline(dt_train[dif], linestyle='--', linewidth=3, color='lime', label='<- train | test ->')
# ax.plot(dt_train, y_raw, color="black", linewidth = 2.5, label="Raw")
ax.plot(dt_train, y_test, color="blue", linewidth = 2.5, label="Conditioned")
ax.plot(dt_train, mu, '--', color="red", linewidth = 2.5, label="GP")
ax.plot(dt_train, yNN, '--', color="orange", linewidth = 2.5, label="NN")
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":12}, facecolor="white", framealpha=1.0)

plt.show()