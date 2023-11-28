import sys
sys.path.append('../..')
sys.path.append('../../..')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as coef
from time import process_time as ptime

from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from DPGP import *

"""
Apply the 3-fold cross-validation analysis to indentify the global GP
hyperparameters. A N_sim number of simulations are run where a DP-GP is
trained with initial hyperparameters sampled from a Uniform distribution.
The negavite log-likelihood, Mean Square Error measure and Coeficient of
determination are saved in a different worsheet for each k-fold simulation
"""

# plt.close('all')

t_start0 = ptime()
# Write each result of the k-fold in a different spreadsheet
for k in range(3):
    
    #-------------------------------------------------------------------------
    # GET THE TRAINING AND VALIDATION DATA FOR EACH K-FOLD
    #-------------------------------------------------------------------------
    
    train_df = pd.read_excel((str(k)+'-fold' + r'.xlsx'),
                                  sheet_name = str(k)+'-train')
    test_df = pd.read_excel((str(k)+'-fold' + r'.xlsx'),
                            sheet_name = str(k)+'-test')
    
    # The training dataset
    X = train_df.loc[:, train_df.columns[:-2]].values
    y = train_df['y'].values
    
    # Standardise the observations (The inputs are already standardised)
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std
    N = len(y)
    
    # Get the datetime at training locations
    dt = train_df['DateTime'].values
    
    # The validation dataset
    X_test = test_df.loc[:, test_df.columns[:-2]].values
    y_test = test_df['y'].values
    dt_test = test_df['DateTime'].values
    N_test = len(y_test)
    
    #-------------------------------------------------------------------------
    # START THE HYPERPARAMETER SAMPLINGS
    #-------------------------------------------------------------------------
    
    # Sample the itial hyperparameters from a uniform distribution
    lower_ls = 0.1
    ls = 12
    std = 0.044
    
    # The SE kernel plus noise variance
    se = 1**2 * RBF(length_scale=ls, length_scale_bounds=(lower_ls, 300))
    wn = WhiteKernel(noise_level=std**2,noise_level_bounds=(1e-5, 1))
    kernel = se + wn
    
    # Train the DP-GP model
    dpgp = DPGP(X, y_norm, init_K=4, kernel=kernel)
    dpgp.train()
        
    # Save results if the model has found a solution, ignore otherwise
    if dpgp.hyperparameters[-1] != 1e-5:
        # Predictions at validation points
        mu_norm = dpgp.predict(X_test)[0]
        
        # Unormalised predictions
        mu = mu_norm*y_std + y_mean
        error = abs(mse(y_test, mu))
        coef_det = coef(y_test, mu)
        
    # Plot the results
    fig, ax = plt.subplots()
    fig.autofmt_xdate()
    # ax.fill_between(dt_test, mu[:,0] + 3*std[:,0], mu[:,0] - 3*std[:,0],
                    # alpha=0.5, color='pink', label='Confidence \nBounds')
    ax.plot(dt, y, color='black')
    ax.plot(dt_test, y_test, color='black')
    ax.plot(dt_test, mu, color='red', linewidth=2,
            label='Distributed\nDP-GP')
    for s in range(3):
        plt.axvline(dt[int(k*s*88)], linestyle='--', linewidth=2,
                    color='blue')
    ax.set_ylim([-0.4, 5])
    ax.set_xlabel('Date-time')
    ax.set_ylabel('Fault density')
    plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

plt.show()
