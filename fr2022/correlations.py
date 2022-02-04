"""Calculate correlations."""

import gspread
import numpy as np
import pandas as pd

csvurl = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTgg1x2P_SS-52MmixvFPtK1JdiEQOeWcvut1Xk65JJFq6KGs0sUxcNADY8LKgc_DK3PEd07S2piyri/pub?gid=0&single=true&output=csv"

# write to GSheet
sheetkey = "1xGJfKlN1UwzoMI71-UMAwlZMDQrKyx7aBgV5_Fn6x78"
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)


data = pd.read_csv(csvurl)

selected = ['Macron', 'Le Pen', 'Pécresse' ,'Zemmour', 'Mélenchon']

pollsters = data['pollster:id'].unique().tolist()
pollsters.remove('volby')

# correlations
wsw = sh.worksheet('correlations')
i = 1
for pollster in pollsters:
    data_pollster = data[data['pollster:id'] == pollster]
    if len(data_pollster.index) > 3:
        corr = data_pollster[selected].corr()
        wsw.update_cell(i, 1,  pollster + ' (' + str(len(data_pollster.index)) + ')')
        wsw.update('B' + str(i), [corr.columns.values.tolist()] + corr.values.tolist())
        wsw.update('A' + str(i + 1), [[x] for x in corr.columns.values.tolist()])
        i = i + len(selected) + 2









# corr = data[selected].corr()
# cov = data.loc[:, selected].cov()
# mean = data.loc[:, selected].mean()

# np.random.multivariate_normal(mean, cov, size=2)



# wsw.update('B1', [corr.columns.values.tolist()] + corr.values.tolist())
# wsw.update('A2', [[x] for x in corr.columns.values.tolist()])
