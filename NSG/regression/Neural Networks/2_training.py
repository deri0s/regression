
import sys
sys.path.insert(0, 'C:\Diego\PhD\Code\phdCode')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as ss
from NSG import data_processing_methods as dpm
from NSG import *

import tensorflow as tf
from tensorflow import keras

# NSG post processes data location
file = 'NSG/regression/Neural Networks/NSG_training_val_data.xlsx'

# Training df
X_df = pd.read_excel(file, sheet_name='X_training')
y_df = pd.read_excel(file, sheet_name='y_training')
y_raw_df = pd.read_excel(file, sheet_name='y_raw_training')
t_df = pd.read_excel(file, sheet_name='time')

# Pre-Process training data
X_train, y_train, N, D, max_lag, time_lags = dpm.align_arrays(X_df, y_df, t_df)
X = ss().fit(X_train).transform(X_train)
y = ss().fit(y_train).transform(y_train)

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
X_train = X[start:end]
y_train = y[start:end]

batchsize = 850
epoch = 400
val_split = 0.15

trained = model.fit(X_train, y_train, batch_size=batchsize, epochs=epoch, verbose=1, validation_split=val_split)

# Model Validation
start = 0
end = 1000
N = start - end
X_test = 
print("Evaluation against the test data \n")
model.evaluate(X_val, y_val_stand)

# Plot model accuracy after each eppch
pd.DataFrame(trained.history).plot(figsize=(8, 5))
plt.title("Accuracy improvement with epochs")

# Save the model
model.save("2HL_68_8")