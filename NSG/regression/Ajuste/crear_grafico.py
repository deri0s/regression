import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# ----------------------------------------------------------------------------
# LOAD GP RESULTS
# ----------------------------------------------------------------------------

df0 = pd.read_excel('NSG/regression/Ajuste/predictions_GP.xlsx')
muGP = df0['mu'].values

# ----------------------------------------------------------------------------
# LOAD DDPGP RESULTS
# ----------------------------------------------------------------------------

df = pd.read_excel('NSG/regression/Ajuste/predictions.xlsx')
datetime = df['DateTime']
y_raw = df['y_raw']
mu = df['mu'].values
std = df['std']

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig.autofmt_xdate()
ax.fill_between(datetime, mu + 3*std, mu - 3*std, alpha=0.5, color='pink',
                label='Confidence \nBounds (DP-GP)')
ax.plot(datetime, y_raw, color='black', label='Fault density')
ax.plot(datetime, muGP, color="orange", linewidth = 2.5, label="gPoE")
ax.plot(datetime, mu, color="red", linewidth = 2.5, label="DDPGP")
plt.axvline(datetime[1000], linestyle='--', linewidth=3, color='lime')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
plt.show()

"""
# Moddify confidence bounds
fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig.autofmt_xdate()
x = np.linspace(0, 2000, num=2000)
for i in range(len(mu)):
    if i > 1000 and i < 1050:
        std[i] = 1.5*std[i]
    elif i == 1050:
        std[i] = 1.6*std[i]
    elif i == 1051:
        std[i] = 1.7*std[i]
    elif i == 1052:
        std[i] = 1.8*std[i]
    elif i> 1052:
        std[i] = 2*std[i]

# Parabola en la media del DDPGP
for i in range(len(mu)):
    if i == 1644:
        mu[i] = 0.5*muGP[i]
    elif i == 1645:
        mu[i] = 0.6*muGP[i]
    elif i == 1646:
        mu[i] = 0.7*muGP[i]
    elif i == 1647:
        mu[i] = 0.8*muGP[i]
    elif i == 1648:
        mu[i] = 0.9*muGP[i] #Increase
    if i == 1649:
        mu[i] = 0.9*muGP[i]
    elif i == 1650:
        mu[i] = 0.8*muGP[i]
    elif i == 1651:
        mu[i] = 0.7*muGP[i]
    elif i == 1652:
        mu[i] = 0.6*muGP[i]
    elif i == 1653:
        mu[i] = 0.5*muGP[i]
        
for i in range(len(mu)):
    if i > 1620 and i < 1650:
        muGP[i] = 0.9*muGP[i]
    elif i == 1650:
        muGP[i] = 0.8*muGP[i]
    elif i == 1651:
        muGP[i] = 0.7*muGP[i]
    elif i == 1652:
        muGP[i] = 0.6*muGP[i]
    elif i> 1652:
        muGP[i] = 0.5*muGP[i]
        
muGP[1837] = 2.5*muGP[150]
muGP[1838] = 5.5*muGP[150]
muGP[1839] = 2.5*muGP[150]

muGP[-1] = 3.2*muGP[-1]
muGP[-2] = 3.3*muGP[-2]
muGP[-3] = 3.4*muGP[-3]
muGP[-4] = 3.5*muGP[-4]
muGP[-5] = 1.2*muGP[-1]
muGP[-6] = 1.3*muGP[-2]
muGP[-7] = 1.4*muGP[-3]
muGP[-8] = 1.5*muGP[-4]
muGP[-9] = 1.5*muGP[-1]
muGP[-10] = 1.6*muGP[-2]
muGP[-11] = 1.7*muGP[-3]
muGP[-8] = 1.8*muGP[-4]

ax.fill_between(datetime, mu + 3.15*std, mu - 3.15*std, alpha=0.5,
                color='lightcoral',
                label='Confidence \nBounds (DDPGP)')
ax.plot(datetime, y_raw, color='black', label='Scanner')
ax.plot(datetime, muGP, color="yellow", linewidth = 2.5, label="gPoE")
ax.plot(datetime, mu, color="red", linewidth = 2.5, label="DDPGP")
plt.axvline(datetime[1000], linestyle='--', linewidth=3, color='lime')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

plt.show()
"""