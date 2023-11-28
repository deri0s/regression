import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('..')
sys.path.append('../..')
from sklearn.decomposition import PCA
import data_processing_methods as dpm

plt.close('all')

"""
Only for the thesis plots
"""

# ----------------------------------------------------------------------------
# LOAD MK4 SCANNER DATA
# ----------------------------------------------------------------------------

# Choose the scanner to read data
scanner = 'MK4'

# Initialise empty data frames
Y_raw_df, Y_df = pd.DataFrame(), pd.DataFrame()

# Loop over available files of post-processed data
for i in range(1, 5):
    file_name = ('../Input Post-Processing ' + str(i) + ' ' +
                 scanner + '.xlsx')

    # The first 3 files are appended to become a single training data-frame
    if i < 4:
        Y_df = Y_df.append(pd.read_excel(file_name,
                                         sheet_name='output_data'))
        Y_raw_df = Y_raw_df.append(pd.read_excel(file_name,
                                   sheet_name='raw_output_data'))

# ----------------------------------------------------------------------------
# PRE-PROCESSING
# ----------------------------------------------------------------------------
max_lag = 2
# This just removes the first max_lag points from the date_time array.
Y_raw = dpm.adjust_time_lag(Y_raw_df['raw_furnace_faults'].values, shift=0,
                            to_remove=max_lag)

# Y_raw standardisation
Y_raw_mean = np.mean(Y_raw_df['raw_furnace_faults'].values)
Y_raw_std = np.std(Y_raw_df['raw_furnace_faults'].values)
Y_raw_stand = dpm.standardise(Y_raw, Y_raw_mean, Y_raw_std)


# Extract corresponding time stamps. Note this essentially just
# removes the first max_lag points from the date_time array.
date_time = dpm.adjust_time_lag(Y_df['Time stamp'].values,
                                shift=0,
                                to_remove=max_lag)

# Select training dataset
start = 3600
end = -1

Y_mk4 = Y_raw[start: end]
dt_train = date_time[start:end]

# ----------------------------------------------------------------------------
# LOAD ISRA-5D SCANNER DATA
# ----------------------------------------------------------------------------

# Choose the scanner to read data
scanner = 'ISRA'

# Initialise empty data frames
Y_raw_df, Y_df = pd.DataFrame(), pd.DataFrame()

# Loop over available files of post-processed data
for i in range(1, 5):
    file_name = ('../Input Post-Processing ' + str(i) + ' ' +
                 scanner + '.xlsx')

    # The first 3 files are appended to become a single training data-frame
    if i < 4:
        Y_df = Y_df.append(pd.read_excel(file_name,
                                         sheet_name='output_data'))
        Y_raw_df = Y_raw_df.append(pd.read_excel(file_name,
                                   sheet_name='raw_output_data'))

# ----------------------------------------------------------------------------
# PRE-PROCESSING
# ----------------------------------------------------------------------------
max_lag = 2
# This just removes the first max_lag points from the date_time array.
Y_raw = dpm.adjust_time_lag(Y_raw_df['raw_furnace_faults'].values, shift=0,
                            to_remove=max_lag)

# Y_raw standardisation
Y_raw_mean = np.mean(Y_raw_df['raw_furnace_faults'].values)
Y_raw_std = np.std(Y_raw_df['raw_furnace_faults'].values)
Y_raw_stand = dpm.standardise(Y_raw, Y_raw_mean, Y_raw_std)


# Extract corresponding time stamps. Note this essentially just
# removes the first max_lag points from the date_time array.
date_time = dpm.adjust_time_lag(Y_df['Time stamp'].values,
                                shift=0,
                                to_remove=max_lag)

Y_isra = Y_raw[start: end]
dt_train_isra = date_time[start:end]

# ----------------------------------------------------------------------------
# PLOT SCANNER DATA
# ----------------------------------------------------------------------------

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

fig.autofmt_xdate()
# ax.plot(dt_train, Y_mk4, color='black', label='MKIV')
ax.plot(dt_train_isra, Y_isra, color="black", label="FS-5DXIN ISRA")
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

plt.show()