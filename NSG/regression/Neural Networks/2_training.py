import sys
sys.path.insert(0, 'C:\Diego\PhD\Code\phdCode')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as ss
from sklearn.model_selection import train_test_split
from NSG import data_processing_methods as dpm
from NSG import *

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping

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
N_train = 1800
X_train, y_train = X[0:N_train], y[0:N_train]
X_test, y_test = X[N_train:N], y[N_train:N]


"""
Neural Network Training
"""
import xlsxwriter
from keras.layers import Dense

# Architecture
architecture = '2HL_32sp_8sp'
act = 'softplus'
model = keras.models.Sequential()
model.add(Dense(32, name='hidden1', activation=act))
model.add(Dense(8, name= 'hidden2', activation=act))
model.add(Dense(1, name= 'output', activation=act))

# Compilation
lr = 0.005
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
              loss='mean_absolute_error')

# Model hyperparameters
B = [900, 600, 300]
epoch = 200
val_split = 0.2

# Patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=940)

# Define an Excel writer object and the target file
location = '\\regression\\Neural Networks\\Varying Hyperparameters\\Batch'
name = 'caca.xlsx'
writer = pd.ExcelWriter(name)

df = []
for i in range(len(B)):
    col = 0
    trained = model.fit(X_train, y_train,
                        batch_size=B[i], epochs=epoch,
                        validation_split=val_split, callbacks=[es])
    
    model.save(architecture+'_'+str(act)+'_B'+str(B[i]))
    
    print("Evaluation against the test data B: ", B[i])
    error = model.evaluate(X_test, y_test)

    # Save 
    d = {}
    d['architecture'] = architecture
    d['B'] = B[i]
    d['Learning rate'] = lr
    d['error'] = error
    df.append(pd.DataFrame(d, index=[0]))

    df[i].to_excel(writer, sheet_name='B '+str(B[i]), index=False)

writer.save()
