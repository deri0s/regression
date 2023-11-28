import sys
sys.path.append('..')
sys.path.append('../../..')
import xlsxwriter
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

# The region of each expert
M = '7.xlsx'

# Create an Excel file where we are going to save the simulation results
workbook = xlsxwriter.Workbook('DPGP k-fold results'+M)

t_start0 = ptime()
# Write each result of the k-fold in a different spreadsheet
for k in [0,1]:
    
    #-------------------------------------------------------------------------
    # INITIALISE WORKSHEETS FOR EACH K-FOLD SIMULATION
    #-------------------------------------------------------------------------
    
    worksheet = workbook.add_worksheet(str(k) + '-fold')
    # Write the titles of the parameters and the Neg-Log-likelihood
    font_title = workbook.add_format({'bold': True, 'font_color': 'blue'})
    worksheet.write(0, 0, 'Initial sf', font_title)
    worksheet.write(0, 1, 'Initial ls', font_title)
    worksheet.write(0, 2, 'Initial std', font_title)
    worksheet.write(0, 3, 'sf', font_title)
    worksheet.write(0, 4, 'ls', font_title)
    worksheet.write(0, 5, 'std', font_title)
    worksheet.write(0, 6, 'Neg Log-likleihood', font_title)
    worksheet.write(0, 7, 'MSE', font_title)
    worksheet.write(0, 8,'R2', font_title)
    
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
    
    N_sim = 275
    for i in range(N_sim):
        print('\n Simulation number: ', i, ' K-fold: ', k)
        col = 0
        
        # Sample the itial hyperparameters from a uniform distribution
        lower_ls = 0.1
        ls = np.random.uniform(lower_ls, 216)
        std = np.random.uniform(1e-3, 1)
        hyper0 = [1, ls, std]
        
        # The SE kernel plus noise variance
        se = 1**2 * RBF(length_scale=ls, length_scale_bounds=(lower_ls, 300))
        wn = WhiteKernel(noise_level=std**2,noise_level_bounds=(1e-5, 2))
        kernel = se + wn
        
        # Train the DP-GP model
        dpgp = DPGP(X, y_norm, init_K=5, kernel=kernel)
        dpgp.train()
        
        # Save results if the model has found a solution, ignore otherwise
        if dpgp.hyperparameters[-1] != 1e-5:
            # Predictions at validation points
            mu_norm = dpgp.predict(X_test)[0]
            
            # Unormalised predictions
            mu = mu_norm*y_std + y_mean
            error = abs(mse(y_test, mu))
            coef_det = coef(y_test, mu)
            
            # Arrange the results in a single array
            elements = (hyper0,
                        np.vstack(dpgp.hyperparameters),
                        [dpgp.log_marginal_likelihood_value_],
                        [error], [coef_det])
            
            # Write the results from the simulation in an Excel file
            for move_elements in range(len(elements)):
                for h in elements[move_elements]:
                    worksheet.write(i+1, col, h)
                    col += 1
                        
    print("K: ", k)
    
time_elapsed = (ptime() - t_start0)
print('Computational time: ', time_elapsed/60, 'minutes')
workbook.close()
