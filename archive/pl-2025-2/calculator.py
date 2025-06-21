"""Calculator for multiple two-races."""

# NOT WORKING FOR MULTIPLE CANDIDATES NOW

# Note: more than 2 candidates is supported: 2 main candidates + "others"

import datetime
import gspread
import math
import numpy as np
import pandas as pd
import scipy.stats
import time
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

sheetkey = "10kUDHS8zh4xxC_7scyhOjHs3ECQSUVKCvVdFXT-IHeY"

path = "pl-2025-2/"

# parameters for the difference
diffpoints = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# load data from GSheet
try:
  gc = gspread.service_account("service_account.json")
except:
  gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)

ws = sh.worksheet('parametry')
dfpreference = pd.DataFrame(ws.get_all_records())

# get additional limits
ws = sh.worksheet('parametry 2')
dflimits = pd.DataFrame(ws.get_all_records())
additional_limits = list(dflimits['additional limits'])

# get number of races:
nraces = 0
for row in dfpreference.iterrows():
  row1 = row[1]
  if row1['name'] != '':
    nraces += 1

# function to get column letter(s) from number
def get_column_letter(n):
    result = ''
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr(65 + remainder) + result
    return result

# for each race:
for nr in range(0, nraces): # last row returns an error
  print(nr)
  # parameters
  election_day = datetime.date.fromisoformat(dfpreference['election date'][nr] if dfpreference['election date'][nr] != "" else dfpreference['election date'][0])
  today = datetime.date.fromisoformat(dfpreference['current date'][nr] if dfpreference['current date'][nr] != "" else dfpreference['current date'][0])
  sample_n = dfpreference['sample n'][nr] if dfpreference['sample n'][nr] != "" else dfpreference['sample n'][0]
  re_coef = dfpreference['re coef'][nr] if dfpreference['re coef'][nr] != "" else dfpreference['re coef'][0]
  aging_coef = dfpreference['aging coef'][nr] if dfpreference['aging coef'][nr] != "" else dfpreference['aging coef'][0]
  sample = dfpreference['sample'][nr] if dfpreference['sample'][nr] != "" else dfpreference['sample'][0]
  interval_min = dfpreference['interval min'][nr] if dfpreference['interval min'][nr] != "" else dfpreference['interval min'][0]
  interval_max = dfpreference['interval max'][nr] if dfpreference['interval max'][nr] != "" else dfpreference['interval max'][0]
  step = dfpreference['step'][nr] if dfpreference['step'][nr] != "" else dfpreference['step'][0]
  volatility = dfpreference['volatilita'][nr] if dfpreference['volatilita'][nr] != "" else dfpreference['volatilita'][0]
  pos = dfpreference.columns.to_list().index('last successful calculation (GMT)')

  # aging curve
  def aging_coeff(day1, day2, coef = 1.15):
    diff = abs((day2 - day1).days)
    if diff <= 0:
      return 1
    return pow(diff, coef) / diff

  # normal error
  def normal_error(p, i, n, volatility, coef = 1):
    # p: a Pandas DataFrame containing data for a set of variables
    # i: an integer specifying the index of the variable within the DataFrame for which the normal error is being calculated
    # n: an integer specifying the sample size
    # volatility: a float specifying the volatility or standard deviation of the variable being measured
    # coef: an optional float specifying to multiply the standard deviation by

    # calculates the standard error
    p['sdx'] = (n * p['p' + str(i)] * (1 - p['p' + str(i)])).apply(abs).apply(math.sqrt) / n * coef * volatility
    # generates a column of normally distributed random values with mean 0 and standard deviation equal to the sdx value for each row
    p['normal_error'] = scipy.stats.norm.rvs(loc=0, scale=p['sdx'])
    return p

  # uniform_error as function of normal error
  def uniform_error(p, i, n, volatility, coef = 1):
    p['sdx'] = (n * p['p' + str(i)] * (1 - p['p' + str(i)])).apply(abs).apply(math.sqrt) / n * coef * volatility
    p['uniform_error'] = scipy.stats.uniform.rvs(loc=(-1 * p['sdx'] * math.sqrt(3)), scale=(2 * p['sdx'] * math.sqrt(3)))
    return p

  # remove empty rows
  dfpreference = dfpreference[dfpreference['name'] != '']
  # create sub-dataframes with only rn row:
  dfpreferencern = dfpreference.iloc[nr: nr + 1]

  # p
  dfpreferencern['p1'] = dfpreferencern['gain1'] / 100
  dfpreferencern['p3'] = 1 - dfpreferencern['p1'] - dfpreferencern['gain2'] / 100

  # simulations
  # simulations = pd.DataFrame(columns=fpreference['name'].to_list())
  simulations_aging = {} 
  for i in range(1, 4):
    simulations_aging[i] = pd.DataFrame(columns=dfpreferencern['name'].to_list())
  aging = aging_coeff(today, election_day, aging_coef)

  for j in range(0, sample):
    for i in {1, 3}:
      p = normal_error(dfpreferencern, i, sample_n, volatility, 1)
      p = uniform_error(p, i, sample_n, volatility, 1.5 * 0.9)
      # p['estimate'] = p['normal_error'] + p['uniform_error'] + p['p']
      p['estimate_aging'] = aging * (p['normal_error'] + p['uniform_error']) + p['p' + str(i)]
      # simx = dict(zip(dfpreference['party'].to_list(), p['estimate']))
      simxa = dict(zip(dfpreferencern['name'].to_list(), p['estimate_aging']))
      # simulations = simulations.append(simx, ignore_index=True)
      simulations_aging[i] = pd.concat([simulations_aging[i], pd.DataFrame([simxa])], ignore_index=True)
      # simulations_aging = simulations_aging.append(simxa, ignore_index=True)

  simulations_aging[2] = 1 - simulations_aging[1] - simulations_aging[3]
  difference12 = simulations_aging[1] - simulations_aging[2]

  # rank
  winning = pd.DataFrame((simulations_aging[1] >= simulations_aging[2]).sum(axis=0) / sample).rename(columns={0: 'p1'})
  winning['p2'] = 1 - winning['p1']

  # write to GSheet
  wsw = sh.worksheet('poradi')
  wsw.update('A' + str(nr + 2), winning.reset_index().values.tolist())
  wsw.update('A1', [['Pr[duel winned]', 'p1', 'p2']])
  # avoid Google Sheets API rate limit
  time.sleep(5)

  # less than
  interval_statistics_aging = {}
  interval = {}
  for j in range(1, 4):
    interval_statistics_aging[j] = pd.DataFrame(columns=dfpreferencern['name'].to_list())
    interval[j] = pd.DataFrame(columns=['Pr[duel zisk > x %]'])
    # for i in np.concatenate((np.arange(0, interval_max + 0.5, 0.5), np.array([26.33, 22.79, 17.11, 9.13, 8.51]))):
    for i in np.concatenate((np.arange(interval_min, interval_max + step, step), np.array(additional_limits))):
      # interval[j] = interval[j].append([{'Pr[duel zisk > x %]': i}], ignore_index=True)
      interval_j_new = pd.DataFrame({'Pr[duel zisk > x %]': [i]})
      interval[j] = pd.concat([interval[j], interval_j_new], ignore_index=True)
      # interval_statistics_aging[j] = interval_statistics_aging[j].append((simulations_aging[j] > (i / 100)).sum() / sample, ignore_index=True)
      interval_statistics_j_new = pd.DataFrame((simulations_aging[j] > (i / 100)).sum() / sample, columns=['interval_statistics_aging'])
      interval_statistics_aging[j] = pd.concat([interval_statistics_aging[j], interval_statistics_j_new], ignore_index=True)
    interval_statistics_aging[j] = interval_statistics_aging[j].loc[:, ['interval_statistics_aging']]

  # write to GSheet
  for j in range(1, 3):
    wsw = sh.worksheet('pravdepodobnosti' + str(j))
    interval[j][interval_statistics_aging[j].columns] = interval_statistics_aging[j]
    interval[j].columns = ['Pr[duel zisk > x %]'] + dfpreferencern['name'].to_list()
    if nr == 0:
      wsw.clear()
      wsw.update('A1', [interval[j].columns.values.tolist()] + interval[j].values.tolist())
    else:
      # write only second column into sheet column B + nr, calculate the name of the column
      column_index = 2 + nr  # B is the 2nd column, so start from 2
      c = get_column_letter(column_index)
      wsw.update(c + '1', [dfpreferencern['name'].to_list()] + interval_statistics_aging[j].values.tolist())

  time.sleep(5)

  # difference 1 - 2 from difference12:
  diffprobs = []
  for dp in diffpoints:
    diffprobs.append(((difference12 > (dp / 100)).sum() / sample).values.tolist())
  # write to GSheet
  wsw = sh.worksheet('difference')
  if nr == 0:
    wsw.clear()
    # save diffpoints to A, values to B
    # diffpoints should be a list of lists
    diffpointsx = [[x] for x in diffpoints]
    wsw.update('A1', [['Pr[difference(X1-X2) > x]']] + diffpointsx)
  column_index = 2 + nr  # B is the 2nd column, so start from 2
  c = get_column_letter(column_index)
  wsw.update(c + '1', [dfpreferencern['name'].to_list()] + diffprobs)
  

  # save history to GSheet, columns date_running	date	match	gain	value	volatilita
  wsw = sh.worksheet('history')
  # save both gains
  for i in range(1, 3):
    # print("i: ", i)
    item = {
      'date_running': datetime.datetime.now().isoformat(),
      'date': dfpreference['election date'][0],
      'match': dfpreferencern['name'][nr],
      'gain': str(i),
      'value': dfpreferencern['gain' + str(i)][nr],
      'volatilita': volatility
    }
    # print("item: ", item)
    # Ensure the data is in the correct format for appending
    row_data = [
        item['date_running'],
        item['date'],
        item['match'],
        item['gain'],
        item['value'],
        item['volatilita']
    ]

    # Append the row to the worksheet
    wsw.append_row(row_data)
  
  # print(dfpreference)  # Check the structure of the DataFrame



# write datetime
wsw = sh.worksheet('parametry')
d = datetime.datetime.now().isoformat()
wsw.update_cell(2, pos + 1, d)