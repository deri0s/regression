import numpy as np
import pandas as pd

def test_post_processed():
    """
    Tests to make sure that all post-processed data has the same columns.
    """

    df1 = pd.read_excel('Input Post-Processing 1 MK4.xlsx',
                        sheet_name='input_data')
    df2 = pd.read_excel('Input Post-Processing 2 MK4.xlsx',
                        sheet_name='input_data')
    df3 = pd.read_excel('Input Post-Processing 3 MK4.xlsx',
                        sheet_name='input_data')
    df4 = pd.read_excel('Input Post-Processing 4 MK4.xlsx',
                        sheet_name='input_data')

    assert all(df1.columns == df2.columns)
    assert all(df1.columns == df3.columns)
    assert all(df1.columns == df4.columns)
