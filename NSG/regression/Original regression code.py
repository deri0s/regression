import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import animation
from sklearn.decomposition import PCA
sys.path.append('..')
sys.path.append('../..')
from LinearReg_Class import LinearReg_Base
from Gen_PoE_GP_Class import Gen_PoE_GP
import data_processing_methods as dpm

"""
Regression models predicting into the future, assuming future
tags are known. Results are saved to a mp4 video.

NOTE:
    MK4:
    We remove the input-output pairs identified at stripe locations to
    ignore faults related to the coating process. We assume the post-processed
    fault density data still contains faults related to the coating process,
    and thus, we ignore them for now.

    ISRA-5D:
    We assume that the data related to the coating process have been
    successfully rectified, and hence we do not ignore any points when
    assembling the training dataset.

Current models:
    - Linear regression with batch training and online updating
    - GP regression with batch training

"""

# ----------------------------------------------------------------------------
# USER OPTIONS
# ----------------------------------------------------------------------------

# Choose the scanner to read data
scanner = 'MK4'

# Furnace properties that are planned (i.e. do not become uncertain as we look
# beyond their corresponding time lags)
planned_inputs = ['10425 Calculated Cullet Ratio',
                  '10091 Furnace Load',
                  '9400 Port 2 Gas Flow (SP)']

# Model inputs to retain (provided by Pilkington, excluding
# combination signals).
to_retain = []
##to_retain.append('10425 Calculated Cullet Ratio')
##to_retain.append('1950 Canal Temp. Control  Pyrometer (2)')
##to_retain.append('10279 Canal Temp. Control (PV)')
to_retain.append('2922 Closed Bottom Temperature - Downstream Working End (PV)')
to_retain.append('2923 Filling Pocket Closed Bottom Temperature Centre (PV)')
to_retain.append('2918 Closed Bottom Temperature - Port 6 (PV)')
to_retain.append('2921 Closed Bottom Temperature - Upstream Working End (PV)')
##to_retain.append('1650 Combustion Air Temperature Measurement')
to_retain.append('10091 Furnace Load')
##to_retain.append('15119 Furnace Pressure (PV)')
##to_retain.append('9393 Glass Level Control (OP)')
to_retain.append('7546 Open Crown Temperature - Port 1 (PV)')
to_retain.append('7746 Open Crown Temperature - Port 2 (PV)')
to_retain.append('7522 Open Crown Temperature - Port 4 (PV)')
to_retain.append('7483 Open Crown Temperature - Port 6 (PV)')
to_retain.append('10271 C9 (T012) Upstream Refiner')
##to_retain.append('7520 Open Crown Temperature - Upstream Working End (PV)')
##to_retain.append('9400 Port 2 Gas Flow (SP)')
##to_retain.append('9282 Tweel Position')
##to_retain.append('11384 Wobbe Index (Incoming Gas)')

'''
# Combined combustion air flow signals
to_retain.append('Port 1 Combustion Air Flow (combined)')
to_retain.append('Port 2 - 3 Combustion Air Flow (combined)')
to_retain.append('Port 4 - 5 Combustion Air Flow (combined)')
to_retain.append('Port 6 - 7 Combustion Air Flow (combined)')
to_retain.append('Port 8 Combustion Air Flow (combined)')

# Combined front wall temperatures
to_retain.append('200000 Front Wall Temperature Average (PV)')

# Combined regenerator crown temperatures
to_retain.append('Regenerator Crown Temperature Port 2 (abs. difference)')
to_retain.append('Regenerator Crown Temperature Port 4 (abs. difference)')
to_retain.append('Regenerator Crown Temperature Port 6 (abs. difference)')
to_retain.append('Regenerator Crown Temperature Port 8 (abs. difference)')

# Combined regenerator base temperatures
to_retain.append('Regenerator Base Temperature Port 1 (abs. difference)')
to_retain.append('Regenerator Base Temperature Port 2 (abs. difference)')
to_retain.append('Regenerator Base Temperature Port 3 (abs. difference)')
to_retain.append('Regenerator Base Temperature Port 4 (abs. difference)')
to_retain.append('Regenerator Base Temperature Port 5 (abs. difference)')
to_retain.append('Regenerator Base Temperature Port 6 (abs. difference)')
to_retain.append('Regenerator Base Temperature Port 7 (abs. difference)')
to_retain.append('Regenerator Base Temperature Port 8 (abs. difference)')

# Other model inputs we can choose to retain if we want
to_retain.append('6539 Bath Pressure (PV)')
to_retain.append('11282 Chimney Draught Pressure - After Ecomomiser')
to_retain.append('5134 EEDS Average Gross Width Measurement')
to_retain.append('11213 Essential Services Board Cat \'B\' Supply - MV53')
to_retain.append('30208 Feeder Speed Measurement Left')
to_retain.append('5288 Front Wall Cooling Air Flow (PV)')
to_retain.append('11201 Furnace & Services Pack Sub S8 L.H. Section - HV13')
to_retain.append('11174 Furnace Bottom Temperature 18m D/S of B8')
to_retain.append('7474 Lehr Drive Line Shaft Speed')
to_retain.append('6463 Main Gas Pressure (PV)')
to_retain.append('11211 MV51')
to_retain.append('7443 Outside Ambient Temperature Measurement')
to_retain.append('7999 Outside Windspeed Anemometer')
to_retain.append('11105 Port 1 Combustion Air Flow LHS (OP)')
to_retain.append('11108 Port 1 Combustion Air Flow RHS (OP)')
to_retain.append('11111 Port 2 - 3 Combustion Air Flow LHS (OP)')
to_retain.append('11114 Port 2 - 3 Combustion Air Flow RHS (OP)')
to_retain.append('11146 Regenerator Base Temperature Port 8 LHS')
to_retain.append('15070 Regenerator Crown Temperature Port 6 RHS')
to_retain.append('11221 Services Building MCC1 - MV67')
to_retain.append('11217 Services Building MCC9 Cat \'B\' Supply - MV60')
to_retain.append('8344 Total CCCW Flow Measurement')
to_retain.append('136 Total Combustion Air Flow Measurement')
to_retain.append('135 Total Firm Gas Flow Measurement')
to_retain.append('30060 U/S Flowing End Air Flow Measurement')
to_retain.append('11301 UK5 Total Load (Power)')
'''
# ----------------------------------------------------------------------------
# LOAD DATA FOR TRANING AND TESTING
# ----------------------------------------------------------------------------

# Initialise empty data frames
X_df, X_df_test = pd.DataFrame(), pd.DataFrame()
Y_df, Y_df_test = pd.DataFrame(), pd.DataFrame()
Y_raw_df, Y_raw_df_test = pd.DataFrame(), pd.DataFrame()

# Loop over available files of post-processed data
for i in range(1, 5):
    file_name = ('../../Input Post-Processing ' + str(i) + ' ' +
                 scanner + '.xlsx')

    # The first 3 files are appended to become a single training data-frame
    if i < 4:
        X_df = X_df.append(pd.read_excel(file_name,
                                         sheet_name='input_data'))
        Y_df = Y_df.append(pd.read_excel(file_name,
                                         sheet_name='output_data'))
        Y_raw_df = Y_raw_df.append(pd.read_excel(file_name,
                                   sheet_name='raw_output_data'))

    # The fourth file is used to create the testing data-frame
    if i == 4:
        X_df_test = X_df_test.append(pd.read_excel(file_name,
                                     sheet_name='input_data'))
        Y_df_test = Y_df_test.append(pd.read_excel(file_name,
                                     sheet_name='output_data'))
        Y_raw_df_test = Y_raw_df_test.append(pd.read_excel(file_name,
                                             sheet_name='raw_output_data'))

        # Extract time lags from final file (should be the same for all)
        T_df = pd.read_excel(file_name, sheet_name='time_lags')

# Check data frames are the correct size and have the same column names
assert np.all(X_df.columns == X_df_test.columns)
assert np.all(X_df.columns == T_df.columns)
assert len(X_df) == len(Y_df)
assert len(Y_df) == len(Y_raw_df)
assert len(X_df_test) == len(Y_df_test)
assert len(Y_df_test) == len(Y_raw_df_test)

# Name of saved video
video_name = scanner + ' test video.mp4'

# Extract stripe locations for the MK4
if scanner == 'MK4':
    stripe_locations = Y_df['Stripe'].values
    stripe_locations_test = Y_df_test['Stripe'].values

# ----------------------------------------------------------------------------
# REMOVE INPUTS WE ARE NOT GOING TO USE
# ----------------------------------------------------------------------------

input_names = X_df.columns
for name in input_names:
    if name not in to_retain:
        X_df.drop(columns=name, inplace=True)
        X_df_test.drop(columns=name, inplace=True)
        T_df.drop(columns=name, inplace=True)

# Check that the data frames contain the correct number of inputs
assert len(X_df.columns) == len(to_retain)

# Check that the data frame input names match those in to_retain
assert set(X_df.columns) == set(to_retain)

# ----------------------------------------------------------------------------
# PRE-PROCESSING
# ----------------------------------------------------------------------------

# Finding fault density mean and std (at training points)
Y_mean = np.mean(Y_df['furnace_faults'].values)
Y_std = np.std(Y_df['furnace_faults'].values)

# Standardise training data
for i in range(np.shape(X_df)[1]):
    tag_name = X_df.columns[i]

    # Get the inputs statistics to use in the training and
    # testing data standardisation
    X_mean = np.mean(X_df.iloc[:, i])
    X_std = np.std(X_df.iloc[:, i])

    # Re-write X_df now with standardise data (at training points)
    X_df[tag_name] = dpm.standardise(X_df.iloc[:, i],
                                     X_mean,
                                     X_std)
    # Re-write X_df_test now with standardise data (at training points)
    X_df_test[tag_name] = dpm.standardise(X_df_test.iloc[:, i],
                                          X_mean,
                                          X_std)

# Standardise testing data
Y_df['furnace_faults'] = dpm.standardise(Y_df['furnace_faults'].values,
                                         Y_mean,
                                         Y_std)

Y_df_test['furnace_faults'] = dpm.standardise(Y_df_test['furnace_faults'].values,
                                              Y_mean,
                                              Y_std)
# Process training data
X, Y, N, D, max_lag, time_lags = dpm.align_arrays(X_df,
                                                  Y_df,
                                                  T_df)

# Process testing data
X_test, Y_test, N_test, D, max_lag, time_lags = dpm.align_arrays(X_df_test,
                                                                 Y_df_test,
                                                                 T_df)

# Process raw target data in the same way as the post-processed
# target data. Note this essentially just removes the first max_lag
# points from the date_time array.
Y_raw = dpm.adjust_time_lag(Y_raw_df['raw_furnace_faults'].values,
                            shift=0,
                            to_remove=max_lag)

Y_raw_test = dpm.adjust_time_lag(Y_raw_df_test['raw_furnace_faults'].values,
                                 shift=0,
                                 to_remove=max_lag)

# Extract corresponding time stamps. Note this essentially just
# removes the first max_lag points from the date_time array.
date_time = dpm.adjust_time_lag(Y_df['Time stamp'].values,
                                shift=0,
                                to_remove=max_lag)

date_time_test = dpm.adjust_time_lag(Y_df_test['Time stamp'].values,
                                     shift=0, to_remove=max_lag)

# Only use the stripe location if we are using the MK4 data
if scanner == 'MK4':

    # Remove the first max_lag points from the stripe
    # locations arrays
    stripe_locations = dpm.adjust_time_lag(stripe_locations, shift=0,
                                           to_remove=max_lag)
    stripe_locations_test = dpm.adjust_time_lag(stripe_locations_test, shift=0,
                                                to_remove=max_lag)

    # Make sure that stripes locations are removed from training data
    training_inx = np.invert(stripe_locations)
    X = X[training_inx]
    Y = Y[training_inx]
    N = len(Y)

# PCA
pca = PCA(n_components=2)
pca.fit(X)
Xt = pca.transform(X)

# PCA on test data
Xt_test = pca.transform(X_test)

# ----------------------------------------------------------------------------
# LINEAR REGRESSION MODEL
# ----------------------------------------------------------------------------


class LinearReg(LinearReg_Base):

    def phi(self, x):
        return np.vstack(x)


lr = LinearReg(X, Y, N, D, lam=1e-3)
print('Linear regression model trained')

# ----------------------------------------------------------------------------
# PRODUCT-OF EXPERTS GAUSSIAN PROCESS
# ----------------------------------------------------------------------------

# Train
initial_theta = [0.1]*(D+1)
gp = Gen_PoE_GP(X, Y, N, N_GPs=20, initial_theta=initial_theta, ARD=True)
gp.train(shared_theta=False)
print('GP regression models trained')

# ----------------------------------------------------------------------------
# FUTURE PREDICTIONS
# ----------------------------------------------------------------------------

N_star = 216   # Inputs 72 hours into the future

# The number of fault density data points that we can see after the model's
# last prediction
plot_lim = 144

# Setting the path for the video writer
animation.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'
animation.rcParams['animation.writer'] = 'ffmpeg'

# Initialise plot for video
fig, ax = plt.subplots(nrows=2, ncols=2)
fig.suptitle('Using '+scanner+' data')
ax = ax.flatten()
fig.delaxes(ax[3])

# Colours to be used in the plot of inputs
colours = cm.gist_rainbow(np.linspace(0, 1, D))


def video_frame(i):
    # Define a frame in the video to be called repeatedly

    print('Frame', str(i), 'out of', str(len(Y_test) - N_star - plot_lim))

    # Input for model predictions
    X_star = X_test[i:i+N_star, :]
    X_star_const = np.copy(X_star)

    # Beyond time lags, replace uncertain inputs with constant values
    for j in range(D):

        # Only replace signal with constants if it has not been
        # specified as a 'planned input'
        if X_df.columns.values[j] not in planned_inputs:
            X_star_const[time_lags[j]:, j] = X_star_const[time_lags[j]-1, j]

    # Make model predictions
    Y_star_lr = lr.predict(X_star_const, N_star)
    Y_star_mu_gp, Y_star_std_gp, beta = gp.predict(X_star_const, N_star)

    # Update first subplot
    indx = range(i, i+N_star)
    ax[0].clear()
    ax[0].plot(date_time_test, Y_raw_test, 'grey',
               label='Raw FD')
    ax[0].plot(date_time_test, Y_test*Y_std + Y_mean, 'black',
               label='Filtered FD')

    if scanner == 'MK4':
        ax[0].plot(date_time_test[stripe_locations_test],
                   Y_test[stripe_locations_test]*Y_std + Y_mean,
                   'o', c='red')

    ax[0].set_title('Fault Density')

    # Plot linear regression predictions
    prediction_line = ax[0].plot(date_time_test[indx],
                                 Y_star_lr*Y_std + Y_mean,
                                 'blue')

    # Plots for MoE GP predictions
    prediction_line_gp = ax[0].plot(date_time_test[indx],
                                    Y_star_mu_gp*Y_std + Y_mean,
                                    'orange')
    ax[0].plot(date_time_test[indx],
               (Y_star_mu_gp+3*Y_star_std_gp)*Y_std + Y_mean, 'yellow')
    ax[0].plot(date_time_test[indx],
               (Y_star_mu_gp-3*Y_star_std_gp)*Y_std + Y_mean, 'yellow')

    # Set axes of first subplot
    ax[0].set_ylim([-0.3, 2])
    ax[0].set_xlim([date_time_test[i - 10],
                    date_time_test[i+N_star + plot_lim]])

    # Project model inputs onto first two principal components
    X_star_const_t = pca.transform(X_star_const)

    # Update second plot for PCA results
    ax[1].clear()
    ax[1].plot(Xt[:, 0], Xt[:, 1], 'o', markersize=0.4, c='green',
               label='Training points', alpha=0.1)
    ax[1].set_xlim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 0])))
    ax[1].set_ylim(np.min(Xt[:, 1]), np.max(np.max(Xt[:, 1])))
    ax[1].plot(X_star_const_t[:, 0], X_star_const_t[:, 1], 'o',
               markersize=0.4, c='black',
               label='Projected future state')
    ax[1].set_title('PCA Plot')
    ax[1].legend()

    # Update third subplot showing model inputs
    ax[2].clear()
    fig.autofmt_xdate()
    input_lines = []

    for j in range(D):

        # Plot how inputs really varied
        input_lines.append(ax[2].plot(date_time_test[indx],
                                      X_star[:, j],
                                      c=colours[j],
                                      label=X_df.columns[j]))

        # Plot inputs actually used for the model
        input_lines.append(ax[2].plot(date_time_test[indx],
                                      X_star_const[:, j],
                                      c=colours[j],
                                      linestyle='--'))

    ax[2].set_xlim([date_time_test[i - 10],
                    date_time_test[i+N_star + plot_lim]])
    ax[2].legend(loc="upper left", bbox_to_anchor=(0.95, 0.95),
                 fontsize='xx-small')

    if scanner == 'MK4':
        # Update parameter estimates only if new observation is not part
        # of a coating period. For now we only updating the parameters of
        # the linear regression model.

        if stripe_locations_test[i]:
            print('Ignoring stripe')
        else:
            lr.sequential_update(X_test[i, :], Y_test[i], eta=1e-2)

    if scanner == 'ISRA':
        lr.sequential_update(X_test[i, :], Y_test[i], eta=1e-2)

    return [prediction_line, prediction_line_gp, input_lines]


# Calling animation function to create a video
lin_ani = animation.FuncAnimation(fig, func=video_frame,
                                  frames=np.arange(10, N_test-N_star-plot_lim),
                                  interval=1, repeat=False)

# Saving the created video
FFwriter = animation.writers['ffmpeg']
lin_ani.save(video_name, writer='ffmpeg', fps=10, metadata={'title': 'test'})
plt.close()
