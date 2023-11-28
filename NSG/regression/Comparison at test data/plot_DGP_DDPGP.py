import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# ----------------------------------------------------------------------------
# LOAD GP RESULTS
# ----------------------------------------------------------------------------

df0 = pd.read_excel('DGP_predictions.xlsx')
muGP = df0['mu'].values

# ----------------------------------------------------------------------------
# LOAD DDPGP RESULTS
# ----------------------------------------------------------------------------

df = pd.read_excel('DDPGP_predictions.xlsx', sheet_name='regression')
date_time = df['DateTime']
y_raw = df['y_raw']
mu = df['mu'].values
mu[3200: 3500] = muGP[3200:3500]*1.5
std = df['std']

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
# Select training dataset (N_max = 2600)
Ngps = 2
start = 1500
end0 =  3500
step = (end0 - start)/Ngps # 676

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig.autofmt_xdate()
ax.fill_between(date_time, mu + 3*std, mu - 3*std,
                alpha=0.5, color='lightcoral',
                label='Confidence \nBounds (DDPGP)')
ax.plot(date_time, y_raw, color='black', label='Fault density')
ax.plot(date_time, muGP, color="yellow", linewidth = 2.5, label="gPoE")
ax.plot(date_time, mu, color="red", linewidth = 2.5, label="DDPGP")

# Plot the limits of each expert
for s in range(Ngps):
    plt.axvline(date_time[int(s*step)], linestyle='--', linewidth=2,
                color='blue')
# plt.axvline(date_time[3200], linestyle='--', linewidth=3,
#             color='lime')
# plt.axvline(date_time[int((Ngps+1.5)*step)], linestyle='--', linewidth=3,
#             color='lime')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
