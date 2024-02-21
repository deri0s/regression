import os
import pandas as pd

root_path = os.getcwd()
data_path = os.path.realpath('NSG/data')

def get_data_path(file_name: str) -> str:
    return os.path.join(data_path, file_name)

# data = pd.read_excel(get_data_path('NSG_data.xlsx'), sheet_name='X_training_stand')
# print(data.head(4))