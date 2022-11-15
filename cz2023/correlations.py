"""Calculate correlations."""

import gspread
import numpy as np
import pandas as pd

# write to GSheet
sheetkey = "1QCOLhcvmC04hiaFikqXGVFTtJ_dttsYxPfS-HQoK6FQ"
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)

ws = sh.worksheet('data copy')
data = pd.DataFrame(ws.get_all_records())

polls_min = 1

selected = ['Petr Pavel', 'Andrej Babiš', 'Danuše Nerudová', 'Marek Hilšer', 'Josef Středula', 'Pavel Fischer']

pollsters = data['pollster:id'].unique().tolist()
# pollsters.remove('volby')

# correlations
wsw = sh.worksheet('correlations')
i = 1
for pollster in pollsters:
  data_pollster = data[data['pollster:id'] == pollster]
  if len(data_pollster.index) >= polls_min:
    corr = data_pollster[selected].corr()
    # fill NAs
    corr.fillna(0, inplace=True)
    corr.values[[np.arange(corr.shape[0])]*2] = 1
    # write to GSheet
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
