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


"""
Changing the batch-size hyperparameter
"""
def fit_model(X, y, D, y0, y_raw, B, epoch, val_split):
    # Architecture
    model = keras.models.Sequential()
    # model.add(keras.layers.Dense(128, name='hidden1', activation="relu", input_dim=D))
    model.add(keras.layers.Dense(32, name='hidden2', activation='relu'))
    model.add(keras.layers.Dense(8, name='hidden3', activation='relu'))
    model.add(keras.layers.Dense(1, name='output', activation='linear'))

    # Compilation
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error')

    # Train and test data
    start = 0
    end = 6000
    dif = 200
    X_train, y_train = X[start:end], y[start:end]
    X_test = X[end-dif:end+dif]
    # y_test doesn't need to be standardise
    y_test = y0[end-dif:end+dif]
    y_raw = y_raw[end-dif:end+dif]
    dt_train = date_time[end-dif:end+dif]
    N = end - start
    assert N > 0

    # Standardise targets
    scaler = ss().fit(np.vstack(y0))
    y = scaler.transform(np.vstack(y0))

    # Patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=9400)

    trained = model.fit(X_train, y_train,
                        batch_size=B, epochs=epoch, verbose=2,
                        validation_split=val_split, callbacks=[es])

    # Plot accuracy of the model after each epoch.
    plt.figure()
    plt.plot(trained.history['loss'], label='train')
    plt.plot(trained.history['val_loss'], label='test')
    plt.title("B="+str(B))
    plt.xlabel("N of Epoch")
    plt.ylabel("Error (MAE)")
    plt.legend()

    # Predictions on test data
    yp_stand = model.predict(X_test)
    yp = scaler.inverse_transform(yp_stand)

    print("Evaluation against the test data \n")
    model.evaluate(X_test, y_test)

    model.save("2HL_32relu_8relu_B3000_epoch10000")

    #Plot accuracy of the model after each epoch.
    fig, ax = plt.subplots()

    # Increase the size of the axis numbers
    plt.rcdefaults()
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)

    fig.autofmt_xdate()
    plt.axvline(dt_train[dif], linestyle='--', linewidth=3, color='lime', label='<- train | test ->')
    ax.plot(dt_train, y_raw, color="black", linewidth = 2.5, label="Raw")
    ax.plot(dt_train, y_test, color="blue", linewidth = 2.5, label="Conditioned")
    ax.plot(dt_train, yp, color="orange", linewidth = 2.5, label="NN")
    ax.set_xlabel(" Date-time", fontsize=14)
    ax.set_ylabel(" Fault density", fontsize=14)
    plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

# How the model error change with B?
# Bs = [32, 64, 300, 500]
Bs = [3000]
epoch = 10000
val_split = 0.2

for i in range(len(Bs)):
    fit_model(X, y, D, y0, y_raw, Bs[i], epoch, val_split)
      
# show learning curves
plt.show()