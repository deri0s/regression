import pandas as pd
import os

path = os.path.realpath('NSG/regression/Neural Networks/')
hl2 = ['4', '32', '64']
for i in range(1, 4):
    if i == 1:
        name = '\\'+str(i)+'HL'
        # df = pd.read_csv(path+name+'.csv')
        df = pd.DataFrame()
    elif i == 2:
        for j in hl2:
            name = '\\'+str(i)+'HL_'+j+'_units'
            df = pd.concat([df, pd.read_excel(path+name+'.xlsx')], ignore_index=True)
    else:
        name = '\\'+str(i)+'HL_32_units'
        df = pd.concat([df, pd.read_excel(path+name+'.xlsx')], ignore_index=True)

print(df)