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
X_df = pd.read_excel(file, sheet_name='X_training')
y_df = pd.read_excel(file, sheet_name='y_training')

y_unstand_df = pd.read_excel(file, sheet_name='y_training_unstand')

y_raw_df = pd.read_excel(file, sheet_name='y_raw_training')

T_df = pd.read_excel(file, sheet_name='time')

# Align STANDARDISED training data
X_train, Y_train, N, D, max_lag, time_lags = dpm.align_arrays(X_df, y_df, T_df)
date_time = dpm.adjust_time_lag(y_df['Time stamp'].values,
                                shift=0,
                                to_remove=max_lag)

# Align UN-STANDARDISED RECTIFIED targets
Y_train_undstand = dpm.adjust_time_lag(y_unstand_df['furnace_faults'].values,
                                       shift=0,
                                       to_remove=max_lag)

# # Align UN-STANDARDISED RAW targets
y_raw = dpm.adjust_time_lag(y_raw_df['raw_furnace_faults'].values,
                            shift=0,
                            to_remove=max_lag)

# Validation dataset
start = 0
end = 15000
X_val = X_train[start:end]
y_val = Y_train[start:end]
y_val_unstand = Y_train_undstand[start:end]
y_raw_val = y_raw[start:end]
dtval = date_time[start:end]

# Load trained model
model = keras.models.load_model('2HL_68_8')
model.summary()
yp = model.predict(X_val)

# Finding fault density mean and std (at training points)
Y_mean = np.mean(y_unstand_df['furnace_faults'].values)
Y_std = np.std(y_unstand_df['furnace_faults'].values)

mu = yp*Y_mean + Y_std

# Plots
plt.figure()
plt.plot(dtval, y_raw_val, 'o', color="black", label="Raw")
plt.plot(dtval, y_val_unstand, color="orange", label="Rectified")
plt.plot(dtval, mu, c='red', label='NN')
plt.legend()
plt.title('NSG')

plt.show()