
import sys
sys.path.insert(0, 'C:\Diego\PhD\Code\phdCode')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from NSG import data_processing_methods as dpm
from NSG import *

import tensorflow as tf
from tensorflow import keras

# NSG post processes data location
file = 'NSG/regression/Neural Networks/NSG_training_val_data.xlsx'

# Training df
X_df = pd.read_excel(file, sheet_name='X_training_stand')
y_df = pd.read_excel(file, sheet_name='y_training_stand')

y_unstand_df = pd.read_excel(file, sheet_name='y_training')

y_raw_df = pd.read_excel(file, sheet_name='y_raw_training')

T_df = pd.read_excel(file, sheet_name='time')

# Validation df
X_val_df = pd.read_excel(file, sheet_name='X_val')

y_val_df = pd.read_excel(file, sheet_name='y_val')

y_raw_val_df = pd.read_excel(file, sheet_name='y_raw_val')

# Process training data
X_train, Y_train, N, D, max_lag, time_lags = dpm.align_arrays(X_df, y_df, T_df)

# Process testing data
X_test, Y_test, N_test, D, max_lag, time_lags = dpm.align_arrays(X_val_df,
                                                                 y_val_df,
                                                                 T_df)

# Process raw target data in the same way as the post-processed
# target data. Note this essentially just removes the first max_lag
# points from the date_time array.
y_raw = dpm.adjust_time_lag(y_raw_df['raw_furnace_faults'].values,
                            shift=0,
                            to_remove=max_lag)
# Y_raw standardisation
Y_raw_mean = np.mean(y_raw_df['raw_furnace_faults'].values)
Y_raw_std = np.std(y_raw_df['raw_furnace_faults'].values)
Y_raw_stand = dpm.standardise(y_raw, Y_raw_mean, Y_raw_std)

# Y_raw_test does not need to be standardise
y_raw_test = dpm.adjust_time_lag(y_raw_val_df['raw_furnace_faults'].values,
                                 shift=0,
                                 to_remove=max_lag)

# Y_raw_test standardisation
Y_raw_mean = np.mean(y_raw_test)
Y_raw_std = np.std(y_raw_test)
y_raw_test_stand = dpm.standardise(y_raw_test, Y_raw_mean, Y_raw_std)

# Extract corresponding time stamps. Note this essentially just
# removes the first max_lag points from the date_time array.
date_time = dpm.adjust_time_lag(y_df['Time stamp'].values,
                                shift=0,
                                to_remove=max_lag)

date_time_test = dpm.adjust_time_lag(y_val_df['Time stamp'].values,
                                     shift=0,
                                     to_remove=max_lag)


"""
Neural Network Training
"""
# Architecture
model = keras.models.Sequential()
N_units1 = 64
model.add(keras.layers.Dense(N_units1, name='hidden1', activation='softplus'))
N_units2 = 8
model.add(keras.layers.Dense(N_units2, name='hidden2', activation='softplus'))
model.add(keras.layers.Dense(1, name='output'))

# Compilation
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss='mean_absolute_error')

# Model training
start = 0
end = 1000
N = start - end
X = X_train[start:end]
y = Y_train[start:end]

batchsize = 850
epoch = 400
val_split = 0.1

trained = model.fit(X, y, batch_size=batchsize, epochs=epoch, verbose=1, validation_split=val_split)

# Prediction on validation dataset just to see the convergence plot
start = 0
end = 500
X_val = X_test[start:end]
y_val_stand = Y_test[start:end]
y_raw_val = y_raw_test[start:end]
dtval = date_time_test[start:end]

print("Evaluation against validation data \n")
model.evaluate(X_val, y_val_stand)

# Plot model accuracy after each eppch
pd.DataFrame(trained.history).plot(figsize=(8, 5))
plt.title("Accuracy improvement with epochs")

# Save the model
model.save("2HL_68_8")