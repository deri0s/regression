# NSG Pilkington Project

## Installation Notes
- Some versions of xlrd don't seem to load xlsx files for some reason but version 1.1.0 works fine.
- Uses matplotlib version 3.1.1
- Uses ffmpeg to save video as mp4 (see https://www.wikihow.com/Install-FFmpeg-on-Windows)

## Version 2
- Now have data processing pipeline that reads from master spreadsheet

## Version 3
- Now processes the data that is affected by reversals.

## Version 4
- minlag is now stored in a meta data sheet of the post-processed spreadsheet
- automatic testing of pushes and pull requests implemented
- now processing and filtering the reversal signals
- videos can now be created automatically using '1_online.py'
- inputs associated with low parameters now automatically removed in '1_online.py'
- 'Model Input Post-Processing 20201201.xlsx' can now be used to realised pipelines 8 and 9 in the slides

## Version 5
- align_arrays method now introduced in data_processing.py
- can now manually vary the time lags before running each model
- current fault density now included as model input
- in 1_online.py the user now has option to save results and/or show video

## Version 6
### 6.0
- Now read from 'UK5 AI Furnace Model Input Pre-Processing' which contains 170 inputs
- Now processes combinations
- Now only lets signals through if signal_valid=1 AND enable = 1
- Corrected error in data_processing.py which caused every input to be the same signal
- Now includes results for Pipeline10_AR
- Time lags currently read directly from 'UK5 AI Furnace Model Input Pre-Processing'
### 6.1
- Corrected error in the automatic removal of inputs associated with low theta values.

## Version 7
- Now loads revised measures of furnace faults
- isolate_time_period method can now be used to process furnace fault signals (was just tag data previously)
- pipeline 10 AR now applied to the revised target

## Version 8
- Tests now load example tag files
- Addressed error in latest xlrd version; Github now loads xlrd version 1.1.0 to do testing
- Now have separate post-processed datasets for validation and training etc.
- Now plots pre-processed fault density in videos

## Version 9
- align_arrays method no longer outputs time lags
- percentage criterion for eliminating thetas from linear regression introduced

## Version 10
### 10.0
- added cross validation for first modelling approach

### 10.1
- Introduced first part of recursive model structure (regularised linear reg)
- Now report relevant inputs for recursive model structure
- Method for randomising time lags (but not yet implemented)

## Version 11
- Removed correlated inputs
- Now have the option for bespoke filtering

## Version 12
- Ignoring Main Gas Pressure.
- We construct the training dataset for the standard LR model (no autoregressive input) using future_pred = 0 in the align_arrays method.
- Standardisation outside the pre-processing pipeline.
- Capturing the statistics of the pre-processed data in the same spreadsheet.
- Standardisation now in the application scripts.
- The scale parameter \theta_0 has been added in the Linear Regression equation
- Data from the 2020 to 2021 period now is available and used for training.

## Version 13
- Corrected align_arrays method
- Added slides describing how align_arrays works
- Auto-Regressive input now optional for the align_arrays method
- 'model1.py' now applied to dataset 3, which is split into training and testing data

## Version 14
- New class for batch and online linear regression
- Future prediction = 0 now keeps all tags
- Batch linear regression script now colours tags depending on the type of variable (manipulated etc.)

## Version 15
- `regression_online.py` now produces plots of inputs multiplied by model parameters
- `data_processing.py` now outputs time stamps alongside furnace fault data. Time stamps are now plotted on the x-axis of plots in `regression_online.py`.
- Including the column "TagID Description" in the "UK5 AI Furnace Model Input Pre-Processing" spreadsheet.
- Only one spreadsheet containing the fault scanner data MK4 and ISRA with all the time periods in one sheet.
- Cleaning code: the user only has to choose the time period required in the data processing file.
- Including both tag data folders, Tag_Data and Tag_Data_2020_2021
- The data processing code automatically generates spreadsheets with "today's" date.

## Version 16
- Inputs that are outside of their engineering limits are now processed in the data-processing-pipeline.
- The upper engineering limit for 6463 Main Gas Pressure has been changed from 1 to 2.
- Online regression now saves results to a spreadsheet that we can look at later.
- Online regression now trains online using raw fault density, with a criterion for ignoring spikes.
- Time lags in the post-processed spreadsheet now saved with tag names in the first row (not column), so consistent with other sheets in the spreadsheet.
- align_arrays no longer takes future_pred as an input as it is no longer needed.
- align_arrays no longer reports eliminated tags as it is no longer needed.
- align_arrays now reads time lags directly from T_df data frame, using tag names.
- Tag321 temporarily removed as the signal is strange in a way that isn't removed by the data-processing pipeline.
- Online regression model now does NOT include a constant term.

## Version 17
- Post processing of online results now plots thetas as well as model inputs
- Can now manually add a list of inputs we want to ignore in online analysis
- Video now plots model inputs
- Now includes clustering analysis
- Furnace faults are now capped to 10 and spikes are removed using sequentially calculated mean and standard deviation
- Order of the low pass filter can now be defined by the user

## Version 18
- Added PCA analysis of MK4 data
- Added method for detecting coating periods
- Coating periods now ignored in online regression analysis
- Auto-Regressive option removed
- Video now plots PCA components of model inputs

## Version 19
- Added Neural Network to the regression script
- Added new approaches to processing the combination signals
- Added new approach to processing the MK4 fault density data
- Added GP with ARD to the regression script

## Version 20
- Added identify_stripes method to data_processing_methods.py
- Stripes in MK4 data are now identified as part of data_processing.py
- Regression model now ignores datapoints associated with stripes
- Fixed test_low_pass_filter method

## Version 21
- Corrected overwritten pre-processing spreadsheet
- Introduced tests for the pre- and post- processing spreadsheets

## Version 22
- Added test data (May 2021)
- Video of the models' regression performance at test points added

## Version 23
- Video now plots highly relevant inputs recognised by ARD, in the order of relevance

## Version 24
- Added the method ```extract_ISRA_modules```
- Added the method ```entropy```
- Relevant inputs identified by the ARD GP are now plotted on the video corresponding to the test data at the right '''date_time'''.
- Inputs are now assumed to be constant beyond their time lags (called 'projected inputs').
- PCA plots now show projected inputs, rather than true inputs.
- Set time lag for 200000 (combined front wall temperature) to 24 hours, instead of 0.

## Version 25
- Add processed ISRA data where high and low values are replaced by average of reamining modules and 
previous readings respectively.
- The video of the models regression performance using the processed ISRA-5D data have been added.
- GP with ARD tries to use 32 bit distance matrix if there's a memory error.
- Now have a single script for realising regression model predictions.
- Adding generalised PoE Gaussian Process

## Version 26
- Data processing script now has option to ignore combination signals
- All input datasets re-processed
- ANN removed from regression script
- Regression script now uses PoE GP
- Tests updated in response to change in post-processed data

## Version 27
- PoE GP now gives confidence bounds
- Names of inputs updated according to feedback from Matt
- PoE GP now allows each GP to have different parameters
- regression script now uses PoE GP where each GP has different parameters 

## Version 28
- Fixed issue with Generalised PoE GP that caused negative beta values to occurr 
- Regression script ready to generate Generalised PoE GP results
- Generalised PoE tests GP updated
