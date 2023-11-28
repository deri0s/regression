import sys
sys.path.append('..')
sys.path.append('../..')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import data_processing_methods as dpm
from datetime import datetime
from matplotlib.pyplot import cm
from sklearn.decomposition import PCA
from Kmeans_Class import Kmeans

"""
...
"""

# ----------------------------------------------------------------------------
# USER OPTIONS
# ----------------------------------------------------------------------------

AR = False   # Determines whether or not we include an auto-regressive input
make_video = True
save_results = False   # Save results to spreadsheet

# Model inputs to retain
to_retain = ['10425 Calculated Cullet Ratio',
             '1950 Canal Temp. Control  Pyrometer (2)',
             '10279 Canal Temp. Control (PV)',
             '2922 Closed Bottom Temperature - Downstream Working End (PV)',
             '2913 Closed Bottom Temperature - Port 1 (PV)',
             '2918 Closed Bottom Temperature - Port 6 (PV)',
             '2921 Closed Bottom Temperature - Upstream Working End (PV)',
             '1650 Combustion Air Temperature Measurement',
             '10091 Furnace Load',
             '15119 Furnace Pressure (PV)',
             '9393 Glass Level Control (OP)',
             '7546 Open Crown Temperature - Port 1 (PV)',
             '7746 Open Crown Temperature - Port 2 (PV)',
             '7673 Open Crown Temperature - Port 5 (PV)',
             '7483 Open Crown Temperature - Port 6 (PV)',
             '10271 Open Crown Temperature - Upstream Refiner (PV)',
             '7520 Open Crown Temperature - Upstream Working End (PV)',
             '9400 Port 2 Gas Flow (SP)',
             '100021 Regenerator Base Temperature Port 1 (combined)',
             '100022 Regenerator Base Temperature Port 2 (combined)',
             '100023 Regenerator Base Temperature Port 3 (combined)',
             '100024 Regenerator Base Temperature Port 4 (combined)',
             '100025 Regenerator Base Temperature Port 5 (combined)',
             '100026 Regenerator Base Temperature Port 6 (combined)',
             '100027 Regenerator Base Temperature Port 7 (combined)',
             '100012 Regenerator Crown Temperature Port 2 (combined)',
             '100018 Regenerator Crown Temperature Port 8 (combined)',
             '9282 Tweel Position',
             '11384 Wobbe Index (Incoming Gas)']

# ----------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------

# Load data frames
file_name = '../../Input Post-Processing 3 2021_04_12.xlsx'
X_df = pd.read_excel(file_name, sheet_name='input_data')
Y_df = pd.read_excel(file_name, sheet_name='output_data')
T_df = pd.read_excel(file_name, sheet_name='time_lags')
Y_raw_df = pd.read_excel(file_name, sheet_name='raw_output_data')
stand_df = pd.read_excel(file_name, sheet_name='Statistics')


# ----------------------------------------------------------------------------
# MANUALLY IGNORE INPUTS
# ----------------------------------------------------------------------------

input_names = X_df.columns
for name in input_names:
    if name not in to_retain:
        X_df.drop(columns=name, inplace=True)
        T_df.drop(columns=name, inplace=True)


# ----------------------------------------------------------------------------
# PRE-PROCESSING
# ----------------------------------------------------------------------------

# Standardise training data
for i in range(np.shape(X_df)[1]):
    tag_name = X_df.columns[i]

    # Re-write X_df now with standardise data (at training points)
    X_df[tag_name] = dpm.standardise(X_df.iloc[:, i],
                                     np.mean(X_df.iloc[:, i]),
                                     np.std(X_df.iloc[:, i]))


# Process data
X, Y, N, D, max_lag  = dpm.align_arrays(X_df, Y_df, T_df, AR)
Y_raw = dpm.adjust_time_lag(Y_raw_df['raw_furnace_faults'].values,
                            shift=0,
                            to_remove=max_lag)

date_time= dpm.adjust_time_lag(Y_df['Time stamp'].values,
                               shift=0,
                               to_remove=max_lag)

# ----------------------------------------------------------------------------
# CLUSTERING
# ----------------------------------------------------------------------------

# PCA
pca = PCA(n_components=2)
pca.fit(X)
Xt = pca.transform(X)

# K-means
K = 7
C = 10* np.random.rand(K, 2)-0.5
kmeans = Kmeans(Xt, C, N, K, Dx=2)
kmeans.train(N_itt=10)

# ----------------------------------------------------------------------------
# PLOTS
# ----------------------------------------------------------------------------

# Plot convergence of k-means
fig, ax = plt.subplots()
ax.plot(kmeans.J_values)
ax.set_xlabel('Iteration')
ax.set_ylabel('J')

# Plot PCA results
fig, ax = plt.subplots()
ax.plot(Xt[:, 0], Xt[:, 1], 'o', alpha=0.1)
ax.plot(kmeans.C[:, 0], kmeans.C[:, 1], 'o', c='black')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

# Plot k-means results
fig, ax = plt.subplots()
for i in range(len(C)):
    indx = np.where(kmeans.Z[:, i]==1)[0]
    name = "Cluster " + str(i+1)
    ax.plot(Xt[indx, 0], Xt[indx, 1], 'o', label=name)
ax.plot(kmeans.C[:, 0], kmeans.C[:, 1], 'o', c='black')
plt.legend()
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

# Plot clusters on time-history data
fig, ax = plt.subplots()
fig.autofmt_xdate()
ax.plot(date_time, Y_raw, 'black')
for i in range(len(C)):
    indx = np.where(kmeans.Z[:, i]==1)[0]
    name = "Cluster " + str(i+1)
    ax.plot(date_time[indx], Y_raw[indx], 'o', label=name)
plt.legend()
ax.set_ylabel('Fault density')

plt.show()
