import paths
import os
import pandas as pd
import numpy as np
from NSG import *

# ----------------------------------------------------------------------------
# USER OPTIONS
# ----------------------------------------------------------------------------

# Choose the scanner where the fault density data is read from
scanner = 'ISRA'

# Choose the furnace inputs
to_retain = ['10091 Furnace Load',
             '10271 C9 (T012) Upstream Refiner',
             '2922 Closed Bottom Temperature - Downstream Working End (PV)',
             '2921 Closed Bottom Temperature - Upstream Working End (PV)',
             '2918 Closed Bottom Temperature - Port 6 (PV)',
             '2923 Filling Pocket Closed Bottom Temperature Centre (PV)',
             '7546 Open Crown Temperature - Port 1 (PV)',
             '7746 Open Crown Temperature - Port 2 (PV)',
             '7522 Open Crown Temperature - Port 4 (PV)',
             '7483 Open Crown Temperature - Port 6 (PV)']

# ----------------------------------------------------------------------------
# LOAD DATA FOR TRANING AND TESTING
# ----------------------------------------------------------------------------

# Initialise empty data frames
X_df, Y_df, Y_raw_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Loop over available files of post-processed data
for i in range(1, 5):
    file_name = paths.get_data_path('Input Post-Processing ' + str(i) + ' ' +
                                    scanner + '.xlsx')
    X_df = X_df._append(pd.read_excel(file_name,
                                      sheet_name='input_data'))
    Y_df = Y_df._append(pd.read_excel(file_name,
                                      sheet_name='output_data'))
    Y_raw_df = Y_raw_df._append(pd.read_excel(file_name,
                                              sheet_name='raw_output_data'))

# Extract time lags from final file
T_df = pd.read_excel(paths.get_data_path('Input Post-Processing 4 ISRA timelags.xlsx'),
                     sheet_name='time_lags')

# Check data frames are the correct size and have the same column names
assert np.all(X_df.columns == T_df.columns)
assert len(X_df) == len(Y_df)
assert len(Y_df) == len(Y_raw_df)

# ----------------------------------------------------------------------------
# REMOVE INPUTS WE ARE NOT GOING TO USE
# ----------------------------------------------------------------------------

input_names = X_df.columns
for name in input_names:
    if name not in to_retain:
        X_df.drop(columns=name, inplace=True)
        T_df.drop(columns=name, inplace=True)

# Check that the data frames contain the correct number of inputs
assert len(X_df.columns) == len(to_retain)

# Check that the data frame input names match those in to_retain
assert set(X_df.columns) == set(to_retain)

# Standardise input data
from sklearn.preprocessing import StandardScaler

# Using StandardScaler produced the same results obtained with the dpm
# standardised method
X_df_stand = StandardScaler().fit_transform(X_df)
X_df = pd.DataFrame(X_df_stand, columns=X_df.columns)

# Save final training and validation data
writer = pd.ExcelWriter(str(os.getcwd())+'\\NSG\\data\\NSG_data.xlsx')

# Save to spreadsheet
X_df.to_excel(writer, sheet_name='X_training_stand', index=False)
Y_df.to_excel(writer, sheet_name='y_training', index=False)
Y_raw_df.to_excel(writer, sheet_name='y_raw_training', index=False)
T_df.to_excel(writer, sheet_name='time', index=False)

writer._save()