"""Calculator for second rounds."""

import datetime
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
  return pow(diff, 1.15) / diff

# normal error
def normal_error(p, n, volatility, coef = 1):
  p['sdx'] = (n * p['p'] * (1 - p['p'])).apply(math.sqrt) / n * coef * volatility
  p['normal_error'] = scipy.stats.norm.rvs(loc=0, scale=p['sdx'])
  return p

# uniform_error as function of normal error
def uniform_error(p, n, volatility, coef = 1):
  p['sdx'] = (n * p['p'] * (1 - p['p'])).apply(math.sqrt) / n * coef * volatility
  p['uniform_error'] = scipy.stats.uniform.rvs(loc=(-1 * p['sdx'] * math.sqrt(3)), scale=(2 * p['sdx'] * math.sqrt(3)))
  return p

# remove empty rows
dfpreference = dfpreference[dfpreference['name'] != '']

# p
dfpreference['p'] = dfpreference['gain1'] / 100

# simulations
# simulations = pd.DataFrame(columns=fpreference['name'].to_list())
simulations_aging = pd.DataFrame(columns=dfpreference['name'].to_list())
aging = aging_coeff(today, election_day)

for i in range(0, sample):
  p = normal_error(dfpreference, sample_n, volatility, 1)
  p = uniform_error(p, sample_n, volatility, 1.5 * 0.9)
  # p['estimate'] = p['normal_error'] + p['uniform_error'] + p['p']
  p['estimate_aging'] = aging * (p['normal_error'] + p['uniform_error']) + p['p']
  # simx = dict(zip(dfpreference['party'].to_list(), p['estimate']))
  simxa = dict(zip(dfpreference['name'].to_list(), p['estimate_aging']))
  # simulations = simulations.append(simx, ignore_index=True)
  simulations_aging = simulations_aging.append(simxa, ignore_index=True)

# rank
winning = pd.DataFrame((simulations_aging >= 0.5).sum(axis=0) / sample).rename(columns={0: 'p1'})
winning['p2'] = 1 - winning['p1']

# write to GSheet
wsw = sh.worksheet('po??ad??')
wsw.update('A1', [winning.reset_index().columns.values.tolist()] + winning.reset_index().values.tolist())
wsw.update_acell('A1', 'Pr[duel winned]')

# less than
interval_statistics_aging = pd.DataFrame(columns=dfpreference['name'].to_list())
interval = pd.DataFrame(columns=['Pr[duel zisk > x %]'])
# for i in np.concatenate((np.arange(0, interval_max + 0.5, 0.5), np.array([26.33, 22.79, 17.11, 9.13, 8.51]))):
for i in np.concatenate((np.arange(interval_min, interval_max + step, step), np.array([]))):
    interval = interval.append({'Pr[duel zisk > x %]': i}, ignore_index=True)
    interval_statistics_aging = interval_statistics_aging.append((simulations_aging > (i / 100)).sum() / sample, ignore_index=True)

# write to GSheet
wsw = sh.worksheet('pravd??podobnosti')
interval[interval_statistics_aging.columns] = interval_statistics_aging
wsw.update('A1', [interval.columns.values.tolist()] + interval.values.tolist())

# write datetime
wsw = sh.worksheet('parametry')
d = datetime.datetime.now().isoformat()
wsw.update_cell(2, pos + 1, d)