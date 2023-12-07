import os
import sys
sys.path.insert(0, 'C:\Diego\PhD\Code\phdCode')
from jax import config

config.update("jax_enable_x64", True)

from jax import jit
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax as ox

print('path: ', os.getcwd())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as ss
from sklearn.model_selection import train_test_split
from NSG import data_processing_methods as dpm
import gpjax as gpx

key = jr.PRNGKey(123)

"""
NSG data
"""
# NSG post processes data location
# file = str(os.getcwd())
file = 'NSG_data.xlsx'

# Training df
X_df = pd.read_excel(file, sheet_name='X_training')
y_df = pd.read_excel(file, sheet_name='y_training')
y_raw_df = pd.read_excel(file, sheet_name='y_raw_training')
t_df = pd.read_excel(file, sheet_name='time')

# Pre-Process training data
X0, y0, N0, D, max_lag, time_lags = dpm.align_arrays(X_df, y_df, t_df)
X = ss().fit(X0).transform(X0)
y = ss().fit(y0).transform(y0)

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

start = 0
end = 1000
X_train, y_train = X[start:end], y[start:end]
X_test, y_test = X[500:1500], y[500:1500]


"""
Gaussian Process application
"""

n_inducing = 50
z = jnp.linspace(-3.0, 3.0, n_inducing).reshape(-1, 1)

fig, ax = plt.subplots()
ax.scatter(X_train, y_train, alpha=0.25, label="Observations", color=cols[0])
ax.plot(X_test, y_test, label="Latent function", linewidth=2, color=cols[1])
ax.vlines(
    x=z,
    ymin=y.min(),
    ymax=y.max(),
    alpha=0.3,
    linewidth=0.5,
    label="Inducing point"
)
ax.legend(loc="best")
plt.show()

# Training
meanf = gpx.mean_functions.Constant()
kernel = gpx.kernels.RBF()
likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
posterior = prior * likelihood

q = gpx.variational_families.CollapsedVariationalGaussian(
    posterior=posterior, inducing_inputs=z
)

elbo = gpx.objectives.CollapsedELBO(negative=True)
print(gpx.cite(elbo))