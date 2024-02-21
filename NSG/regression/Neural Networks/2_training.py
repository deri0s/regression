import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from NSG import data_processing_methods as dpm
from NSG import *
import paths

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping

# ! Best configuration 3HL-64, 32, 8, units, B=5518, lr=1e-4, epoch=3000, val-split=0.15
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

# Extract corresponding time stamps. Note this essentially just
# removes the first max_lag points from the date_time array.
date_time = dpm.adjust_time_lag(y_df['Time stamp'].values,
                                shift=0,
                                to_remove=max_lag)

# Train and test data
N, D = np.shape(X)
N_train = N
stand = 1

# Scale data using MinMaxScaler. Do not use StandardScaler since the NN training stage
# struggles to find a solution.
scaler = MinMaxScaler(feature_range=(0,1))

def get_train_test(X: np.array, y_orig: np.array, stand: bool):
    y = scaler.fit_transform(np.vstack(y0))

    if stand:
        X_train, y_train = X[0:N_train], y[0:N_train]
        X_test, y_test = X[0:N], y_orig[0:N]
        label = 'Standardised'
    else:
        X_train, y_train = X[0:N_train], y_orig[0:N_train]
        X_test, y_test = X[0:N], y_orig[0:N]
        label = 'Nonstandardised'

    return X_train, y_train, X_test, y_test, label

X_train, y_train, X_test, y_test, stand_label = get_train_test(X, y0, stand)
date_time = date_time[0:N]


"""
Neural Network Training
"""
from keras.layers import Dense, Dropout
from sklearn.metrics import mean_absolute_error as mae

# get best configuration from the single layer analysis
path = os.path.realpath('NSG/regression/Neural Networks')
units = [1024, 256, 128, 64, 32, 8, 2]

N_layers = 3
N_units = units[3]
architecture = '\\'+str(N_layers)+'HL_'+str(N_units)+'_units_'
act = 'relu'
model = keras.models.Sequential()
model.add(Dense(N_units, name='hidden1', activation=act,
                kernel_initializer=tf.keras.initializers.HeNormal()))
model.add(Dropout(0.2))
model.add(Dense(units[4], name='hidden2', activation=act))
model.add(Dropout(0.2))
model.add(Dense(units[5], name='hidden3', activation=act))
model.add(Dense(1, name= 'output', activation='linear'))

# Compilation
lr = 0.0001
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr),
              loss='mean_absolute_error')

# Model hyperparameters
# B = [11036, 5518, 2759, 1379, 690, 345, 172, 64, 32]
B = [32]
epoch = 3000
val_split = 0.15

# Patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=500)

# Define an Excel writer object and the target file
cd = os.getcwd()+'\\NSG\\regression\\Neural Networks'
name_model = cd+architecture+stand_label

for i in range(len(B)):
    print('B: ', B[i])
    trained = model.fit(X_train, y_train,
                        batch_size=int(B[i]), epochs=epoch,
                        validation_split=val_split, verbose=0, callbacks=[es])
    
    model.save(name_model+'_'+str(act)+'_B'+str(B[i]))

    # Plot accuracy of the model after each epoch.
    plt.figure()
    plt.plot(trained.history['loss'], label='train')
    plt.plot(trained.history['val_loss'], label='test')
    plt.title(stand_label+" Units="+str(N_units)+" B="+str(B[i]))
    plt.xlabel("N of Epoch")
    plt.ylabel("Error (MAE)")
    plt.legend()
    
    print("Evaluation against the test data B: ", B[i])
    error = model.evaluate(X_test, y_test)

    # Predictions on test data
    yNN = model.predict(X_test, verbose=0)

    if stand_label:
        yNN = scaler.inverse_transform(yNN)

    yNN_test = yNN[-int(N_train*val_split):]
    error_test = mae(y_test[-int(N_train*val_split):], yNN_test)
    print('\nMAE test:', error_test)

#     # Save
#     d = {}
#     d['architecture'] = architecture
#     d['B'] = B[i]
#     d['Learning rate'] = lr
#     d['error'] = error
#     d['error_test'] = error_test

#     if i == 0:
#         df = pd.DataFrame(d, index=[0])
#     else:
#         df = pd.concat([df, pd.DataFrame(d, index=[0])], ignore_index=True)

# # plt.show()
# df.to_csv(name_model+'_'+str(act)+'_B'+str(B[0])+'.csv', index=False)
print('sali del loop y guarde')
"""
Plots
"""
fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig.autofmt_xdate()
ax.plot(date_time, y_raw, color="grey", linewidth = 2.5, label="Raw")
ax.plot(date_time, y0, color="blue", linewidth = 2.5, label="Conditioned")
ax.plot(date_time, yNN, '--', color="red", linewidth = 2.5, label="NN")
plt.fill_between(date_time[N_train-int(N_train*val_split):], 50, color='pink', label='test data')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":12}, facecolor="white", framealpha=1.0)

plt.show()