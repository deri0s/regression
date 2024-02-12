import os
import pandas as pd

root_path = os.getcwd()
data_path = os.path.realpath('NSG/data')

def get_data_path(file_name: str) -> str:
    return os.path.join(data_path, file_name)

# data = pd.read_excel(file_path, sheet_name='X_training')
# print(data.head(4))