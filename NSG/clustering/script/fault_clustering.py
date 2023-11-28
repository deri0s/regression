import sys
sys.path.append('..')
sys.path.append('../..')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import math

"""
...
"""

# State the columns that we want to retain
retain_classes = False
retain_modules = True
to_retain = []

if retain_classes:
    to_retain += ['Class2Faults',
                  'Class3Faults',
                  'Class4Faults',
                  'Class5Faults',
                  'Class6Faults',
                  'Class7Faults',
                  'Class8Faults',
                  'Class9Faults',
                  'Class10Faults',
                  'Class11Faults',
                  'Class12Faults',
                  'Class13Faults',
                  'Class14Faults',
                  'Class15Faults']

if retain_modules:
    to_retain += ['Module1Faults',
                  'Module2Faults',
                  'Module3Faults',
                  'Module4Faults',
                  'Module5Faults',
                  'Module6Faults',
                  'Module7Faults',
                  'Module8Faults',
                  'Module9Faults',
                  'Module10Faults',
                  'Module11Faults',
                  'Module12Faults',
                  'Module13Faults',
                  'Module14Faults',
                  'Module15Faults']


# Load MK4 data
file_name = '../../Fault Scanner Data 20190801 to 20210218.xlsx'
MK4 = pd.read_excel(file_name, sheet_name='MK4')

# Plot time histories of separate values
fig, ax = plt.subplots(nrows=2, ncols=math.ceil(len(to_retain)/2))
fig.autofmt_xdate()
ax_fl = ax.flatten()
for i in range(len(to_retain)):
    ax_fl[i].plot(MK4['DateTime'], MK4[to_retain[i]].values)
    ax_fl[i].set_title(to_retain[i])

# PCA
X = MK4[to_retain].values
pca = PCA(n_components=2)
pca.fit(X)
Xt = pca.transform(X)

# Outliers
i = np.where(np.abs(Xt[:, 0]) > 50)[0]
i = np.append(i, np.where(np.abs(Xt[:, 1]) > 50)[0])

# Plot PCA results
fig, ax = plt.subplots()
ax.plot(Xt[:, 0], Xt[:, 1], 'o', alpha=0.5)
ax.plot(Xt[i, 0], Xt[i, 1], 'o')

# Plot time history
fig, ax = plt.subplots()
fig.autofmt_xdate()
ax.plot(MK4['DateTime'].values, MK4['TotalFaults'].values)
ax.plot(MK4['DateTime'].values[i], MK4['TotalFaults'].values[i], 'o')

plt.show()
