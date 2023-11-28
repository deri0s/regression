import pandas as pd
import numpy as np

"""
    Create a spreadsheet where the columns corresponds to the relevant inputs,
    and the rows are the timelags sampled from a Gaussian distribution whose
    mean is the timelags estimated by Sarini
"""

# Choose the furnace inputs
relevant = ['10091 Furnace Load',
             '10271 C9 (T012) Upstream Refiner',
             '2922 Closed Bottom Temperature - Downstream Working End (PV)',
             '2921 Closed Bottom Temperature - Upstream Working End (PV)',
             '2918 Closed Bottom Temperature - Port 6 (PV)',
             '2923 Filling Pocket Closed Bottom Temperature Centre (PV)',
             '7546 Open Crown Temperature - Port 1 (PV)',
             '7746 Open Crown Temperature - Port 2 (PV)',
             '7522 Open Crown Temperature - Port 4 (PV)',
             '7483 Open Crown Temperature - Port 6 (PV)']

# Extract time lags
T_df = pd.read_excel('../../Input Post-Processing 4 ISRA timelags.xlsx',
                     sheet_name='time_lags')

timelags_df = pd.DataFrame()

# Get only the relevant inputs
input_names = T_df.columns
for name in input_names:
    if name not in relevant:
        T_df.drop(columns=name, inplace=True)

# # Up to 6 hours from the mean timelag
# N_samples = 200
# std = 18
# d = {T_df.columns[0]:np.random.normal(T_df.iloc[0,0], std, 200)}

# for i in range(1, len(T_df.columns)):
#     d[T_df.columns[i]] = np.random.normal(T_df.iloc[0,i], std, 200)
    
# samples = pd.DataFrame(d)

# # Define an Excel writer object and the target file
# writer = pd.ExcelWriter('Timelag_samples.xlsx')

# # Save to spreadsheet
# samples.to_excel(writer, sheet_name='time_lags', index=False)
# writer.save()