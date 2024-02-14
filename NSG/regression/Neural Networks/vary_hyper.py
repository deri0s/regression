import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as ss
from NSG import data_processing_methods as dpm
from NSG import *
import paths

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping

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
X_train, y_train = X[0:N_train], y0[0:N_train]
X_test, y_test = X[0:N], y0[0:N]


"""
Neural Network Training
"""
import xlsxwriter
from keras.layers import Dense

# Architecture
N_layers = [1]
# N_layers = np.arange(1,6)
N_units = [i*256 for i in range(1,10)][::-1]
act = 'relu'
architecture = '\\'+str(N_layers)+'HL'+str(max(N_units))+'_units_'+act
model = keras.models.Sequential()

# Model hyperparameters
B = [int(N/i) for i in range(1,6)]
epoch = 6000
val_split = 0.2
lr = 1e-4

# Patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=3500)

err = []
bat = []
uni = []
for layer in N_layers:
    for unit in N_units:
        model.add(Dense(unit, name='hidden_'+str(layer), activation=act))
        model.add(Dense(1, name= 'output', activation='linear'))
        model.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr),
                      loss='mean_absolute_error')

        for i in B:
            trained = model.fit(X_train, y_train,
                                batch_size=i, epochs=epoch,
                                validation_split=val_split, callbacks=[es])
            # assemble DataFrame
            err.append(model.evaluate(X_test, y_test))
            bat.append(i)
            uni.append(unit)

        # delete model
        model = keras.models.Sequential()

# assemble DataFrame
df = pd.DataFrame({'units': uni, 'batch': bat, 'MAE': err})
df.to_csv('single_layer.csv')

# Define an Excel writer object and the target file
cd = os.getcwd()    # Current directory
location = '\\regression\\Neural Networks\\Varying Hyperparameters\\Batch'
name = architecture+'_lr_p001'+'.xlsx'
writer = pd.ExcelWriter(cd+location+name)


model.save(cd+location+architecture+'_'+str(act)+'_B'+str(B[i]))

# Plot accuracy of the model after each epoch.
plt.figure()
plt.plot(trained.history['loss'], label='train')
plt.plot(trained.history['val_loss'], label='test')
plt.title("B="+str(B[i]))
plt.xlabel("N of Epoch")
plt.ylabel("Error (MAE)")
plt.legend()

print("Evaluation against the test data B: ", B[i])
error = model.evaluate(X_test, y_test)

    # Save 
    # d = {}
    # d['architecture'] = architecture
    # d['B'] = B[i]
    # d['Learning rate'] = lr
    # d['error'] = error
    # df.append(pd.DataFrame(d, index=[0]))

    # df[i].to_excel(writer, sheet_name='B '+str(B[i]), index=False)

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
# plt.axvline(date_time[N_train], linestyle='--', linewidth=3, color='lime', label='<- train | test ->')
ax.plot(date_time, y_raw, color="grey", linewidth = 2.5, label="Raw")
ax.plot(date_time, y_train, color="blue", linewidth = 2.5, label="Conditioned")
ax.plot(date_time, yNN, '--', color="red", linewidth = 2.5, label="NN")
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":12}, facecolor="white", framealpha=1.0)

plt.show()
# writer._save()
print('caca')