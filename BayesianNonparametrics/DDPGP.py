import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from BayesianNonparametrics.DPGP import DirichletProcessGaussianProcess as DPGP
import matplotlib.pyplot as plt

"""
A distributed robust GP class where each expert is a DP-GP model. 
Diego Echeverria
"""


class DistributedDPGP(GPR):

    def __init__(self, X, Y, N_GPs, init_K, kernel, normalise_y=False):
        """
            Initialise objects, variables and parameters
        """
        
        # Initialise a GP class to access the kernel object when inheriting
        # from the DGP class
        super().__init__(kernel=kernel, alpha=0,
                         normalize_y = normalise_y, n_restarts_optimizer = 0)

        self.X = np.vstack(X)  # Inputs always vstacked
        self.Y = np.vstack(Y)  # Targets always vstacked
        self.N = len(Y)        # No. training points
        self.N_GPs = N_GPs     # No. GPs
        
        # The upper bound of the number of Gaussian noise sources
        self.init_K = init_K
        self.kernel = kernel
        self.normalise_y = normalise_y
        self.independent_hyper = isinstance(self.kernel, list)
        self.D = np.shape(self.X)[1]  # Dimension of inputs

        # Divide up data evenly between GPs
        self.X_split = np.array_split(self.X, N_GPs)
        self.Y_split = np.array_split(self.Y, N_GPs)
        
        # Array of colors to use in the plots
        self.c = ['red', 'orange', 'blue', 'black', 'green',
                  'cyan', 'darkred', 'pink', 'gray', 'magenta',
                  'lightgreen', 'darkblue', 'yellow']
        
        
    def train(self, tol=12):
        """
        Description
        -----------
            The Product of Robust GP experst training
        """

        rgps = []
        for m in range(self.N_GPs):
            # Check if independent hyperparameters have been given
            if self.independent_hyper:
                rgp = DPGP(self.X_split[m], self.Y_split[m],
                           init_K=self.init_K, kernel=self.kernel[m],
                           normalise_y=self.normalise_y)
                rgp.train(tol)
                
                # Print the optimal hyperparameters
                print('\n Expert ', m, " trained")
                print('Hyper -> ', rgp.kernel_, '\n')
                rgps.append(rgp)
            else:
                # All the RGP experts share the same hyperparameters
                rgp = DPGP(self.X_split[m], self.Y_split[m],
                           init_K=self.init_K, kernel=self.kernel)
                rgp.train()
                # Print the optimal hyperparameters
                print('\n Expert ', m, " trained")
                print('Hyper exper: -> ', rgp.kernel_, '\n')
                rgps.append(rgp)
                
        self.rgps = rgps


    def predict(self, X_star):
        """
        Description
        -----------
            Prediction tasks for the rBCM model
        Parameters
        ----------
            X_star : numpy array of new inputs
            N_star : no. new inputs
        Returns
        -------
            mu_star : vector of the rBCM predicitve mean values
            sigma_star : vector of the rBCM predicitve std values
            beta : [N_star x N_GPs] predictive power of each expert
        """
        
        # Collect the experts predictive mean and standard deviation
        N_star = len(X_star)
        mu_all = np.zeros([N_star, self.N_GPs])
        sigma_all = np.zeros([N_star, self.N_GPs])
        
        # Compute local predictions if X_test is > 4000
        if len(X_star) > 4000:
            
            # Divide the test input space into 10 sub-spaces
            X_star_split = np.array_split(X_star, 10)
            N_local = len(X_star_split)
            
            # Outer loop: Move only on the GPs
            for i in range(self.N_GPs):
                full_mean_exp = []
                full_std_exp = []                
            
                # Inner loop: Move only on X_star_split
                for k in range(N_local):
                    mu, sigma = self.rgps[i].predict(X_star_split[k])
                    full_mean_exp.extend(mu[:, 0])
                    full_std_exp.extend(sigma[:, 0])
                    
                mu_all[:, i] = np.asarray(full_mean_exp)
                sigma_all[:, i] = np.asarray(full_std_exp)
        else:
            fig, ax = plt.subplots()
            x_plot = np.linspace(0,self.X[-1], len(self.X))
            x_star_plot = np.linspace(0, X_star[-1], len(X_star))
            step = int(len(self.X)/self.N_GPs)
            advance = 0
            for i in range(self.N_GPs):
                mu, sigma = self.rgps[i].predict(X_star)
                mu_all[:, i] = mu[:, 0]
                sigma_all[:, i] = sigma[:, 0]
                
                # Plot the predictions of each expert
                print('\n Advance: ', advance)
                ax.plot(x_star_plot, mu, color=self.c[i])
                plt.axvline(x_plot[int(advance)], linestyle='--', linewidth=3,
                            color='black')              
                advance += step

        # Calculate the normalised predictive power of the predictions made
        # by each GP. Note that, we are assuming that k(x_star, x_star)=1
        betas = np.zeros([N_star, self.N_GPs])
        # Add Jitter term to prevent numeric error
        prior_std = 1 + 1e-6
        # betas
        for i in range(self.N_GPs):
            betas[:, i] = 0.5*(np.log(prior_std) - np.log(sigma_all[:, i]**2))

        # Normalise beta
        for i in range(N_star):
            betas[i, :] = betas[i, :] / np.sum(betas[i, :])

        # Compute the rBCM precision
        prec_star = np.zeros(N_star)
        for i in range(self.N_GPs):
            prec_star += betas[:, i] * sigma_all[:, i]**-2

        # Compute the rBCM predictive variance and standard deviation
        var_star = prec_star**-1
        std_star = var_star**0.5

        # Compute the rBCM predictive mean
        mu_star = np.zeros(N_star)
        for i in range(self.N_GPs):
            mu_star += betas[:, i] * sigma_all[:, i]**-2 * mu_all[:, i]
        mu_star *= var_star

        return np.vstack(mu_star), np.vstack(std_star), np.vstack(betas)