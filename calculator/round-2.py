"""Calculator for second rounds."""

# Note: more than 2 candidates is supported: 2 main candidates + "others"

import datetime
import gspread
import math
import numpy as np
import pandas as pd
import scipy.stats

sheetkey = "1hrX-5-LORD3jFjeLK51ZxzNW_1OReZvwqh5do4A1R68"

path = "calculator/"

# load data from GSheet
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)

ws = sh.worksheet('parametry')
dfpreference = pd.DataFrame(ws.get_all_records())

# paramenters
election_day = datetime.date.fromisoformat(dfpreference['election date'][0])
today = datetime.date.fromisoformat(dfpreference['current date'][0])
sample_n = dfpreference['sample n'][0]
re_coef = dfpreference['re coef'][0]
aging_coef = dfpreference['aging coef'][0]
sample = dfpreference['sample'][0]
interval_min = dfpreference['interval min'][0]
interval_max = dfpreference['interval max'][0]
step = dfpreference['step'][0]
volatility = dfpreference['volatilita'][0]
pos = dfpreference.columns.to_list().index('last successful calculation (GMT)')

# aging curve
def aging_coeff(day1, day2):
  diff = abs((day2 - day1).days)
  if diff <= 0:
    return 1
  return pow(diff, 1.15) / diff

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

# p
dfpreference['p1'] = dfpreference['gain1'] / 100
dfpreference['p3'] = 1 - dfpreference['p1'] - dfpreference['gain2'] / 100

# simulations
# simulations = pd.DataFrame(columns=fpreference['name'].to_list())
simulations_aging = {}
for i in range(1, 4):
  simulations_aging[i] = pd.DataFrame(columns=dfpreference['name'].to_list())
aging = aging_coeff(today, election_day)

for j in range(0, sample):
  for i in {1, 3}:
    p = normal_error(dfpreference, i, sample_n, volatility, 1)
    p = uniform_error(p, i, sample_n, volatility, 1.5 * 0.9)
    # p['estimate'] = p['normal_error'] + p['uniform_error'] + p['p']
    p['estimate_aging'] = aging * (p['normal_error'] + p['uniform_error']) + p['p' + str(i)]
    # simx = dict(zip(dfpreference['party'].to_list(), p['estimate']))
    simxa = dict(zip(dfpreference['name'].to_list(), p['estimate_aging']))
    # simulations = simulations.append(simx, ignore_index=True)
    simulations_aging[i] = pd.concat([simulations_aging[i], pd.DataFrame([simxa])], ignore_index=True)
    # simulations_aging = simulations_aging.append(simxa, ignore_index=True)

simulations_aging[2] = 1 - simulations_aging[1] - simulations_aging[3]

# rank
winning = pd.DataFrame((simulations_aging[1] >= simulations_aging[2]).sum(axis=0) / sample).rename(columns={0: 'p1'})
winning['p2'] = 1 - winning['p1']

# write to GSheet
wsw = sh.worksheet('pořadí')
wsw.update('A1', [winning.reset_index().columns.values.tolist()] + winning.reset_index().values.tolist())
wsw.update_acell('A1', 'Pr[duel winned]')

# less than
interval_statistics_aging = {}
interval = {}
for j in range(1, 4):
  interval_statistics_aging[j] = pd.DataFrame(columns=dfpreference['name'].to_list())
  interval[j] = pd.DataFrame(columns=['Pr[duel zisk > x %]'])
  # for i in np.concatenate((np.arange(0, interval_max + 0.5, 0.5), np.array([26.33, 22.79, 17.11, 9.13, 8.51]))):
  for i in np.concatenate((np.arange(interval_min, interval_max + step, step), np.array([]))):
    # interval[j] = interval[j].append([{'Pr[duel zisk > x %]': i}], ignore_index=True)
    interval_j_new = pd.DataFrame({'Pr[duel zisk > x %]': [i]})
    interval[j] = pd.concat([interval[j], interval_j_new], ignore_index=True)
    # interval_statistics_aging[j] = interval_statistics_aging[j].append((simulations_aging[j] > (i / 100)).sum() / sample, ignore_index=True)
    interval_statistics_j_new = pd.DataFrame((simulations_aging[j] > (i / 100)).sum() / sample, columns=['interval_statistics_aging'])
    interval_statistics_aging[j] = pd.concat([interval_statistics_aging[j], interval_statistics_j_new], ignore_index=True)
  interval_statistics_aging[j] = interval_statistics_aging[j].loc[:, ['interval_statistics_aging']]

# write to GSheet
for j in range(1, 3):
  wsw = sh.worksheet('pravděpodobnosti' + str(j))
  interval[j][interval_statistics_aging[j].columns] = interval_statistics_aging[j]
  wsw.update('A1', [interval[j].columns.values.tolist()] + interval[j].values.tolist())

# write datetime
wsw = sh.worksheet('parametry')
d = datetime.datetime.now().isoformat()
wsw.update_cell(2, pos + 1, d)