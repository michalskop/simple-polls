"""Simulations for FR 2022. 2nd round"""

from copy import copy
import datetime
import gspread
import math
import numpy as np
import pandas as pd
import scipy.stats
# from matplotlib import pyplot as plt

election_date = '2022-04-24'
election_day = datetime.date.fromisoformat(election_date)
today = datetime.date.today()   # it changes later !!!
sample_n = 1000 # used in statistical error
re_coef = 0.6 # random error coefficient
sample = 10000 # number of simulation
interval_min = 30 # lowest gain to calc probability
interval_max = 70 # highest gain to calc probability
# source sheet
sheetkey = "1c_-hEVNYg-9Boha7S0LV0x2-VY9kKTncNjlHnZI-j2k"
sheetkey1 = "1xGJfKlN1UwzoMI71-UMAwlZMDQrKyx7aBgV5_Fn6x78"
path = "fr2022/"

# load data from GSheet
gc = gspread.service_account()

sh = gc.open_by_key(sheetkey)

ws = sh.worksheet('preference, ze kterých se to počítá')
dfpreference = pd.DataFrame(ws.get_all_records())
dfpreference['p'] = dfpreference['gain1'] / 100
# today
today = datetime.date.fromisoformat(dfpreference['date'][0])

# aging curve 
def aging_coeff(day1, day2):
    diff = abs((day2 - day1).days)
    return pow(diff, 1.15) / diff

# p = dfpreference
# n = sample_n
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

# simulations
simulations = pd.DataFrame(columns=dfpreference['party'].to_list())
simulations_aging = pd.DataFrame(columns=dfpreference['party'].to_list())
aging = aging_coeff(today, election_day)

for i in range(0, sample):
    p = normal_error(dfpreference, sample_n, dfpreference['volatilita'], 1)
    p = uniform_error(p, sample_n, dfpreference['volatilita'], 1.5 * 0.9)
    p['estimate'] = p['normal_error'] + p['uniform_error'] + p['p']
    p['estimate_aging'] = aging * (p['normal_error'] + p['uniform_error']) + p['p']
    simx = dict(zip(dfpreference['party'].to_list(), p['estimate']))
    simxa = dict(zip(dfpreference['party'].to_list(), p['estimate_aging']))
    simulations = simulations.append(simx, ignore_index=True)
    simulations_aging = simulations_aging.append(simxa, ignore_index=True)

# rank
winning = pd.DataFrame((simulations_aging >= 0.5).sum(axis=0) / sample).rename(columns={0: 'p1'})
winning['p2'] = 1 - winning['p1']
winning_now = pd.DataFrame((simulations >= 0.5).sum(axis=0) / sample).rename(columns={0: 'p1'})
winning_now['p2'] = 1 - winning_now['p1']

# write to GSheet
wsw = sh.worksheet('pořadí_aktuální_aging')
wsw.update('A1', [winning.reset_index().columns.values.tolist()] + winning.reset_index().values.tolist())
wsw.update_acell('A1', 'Pr[duel winned]')

wsw = sh.worksheet('pořadí_aktuální')
wsw.update('A1', [winning_now.reset_index().columns.values.tolist()] + winning_now.reset_index().values.tolist())
wsw.update_acell('A1', 'Pr[duel winned]')

# less than
interval_statistics_aging = pd.DataFrame(columns=dfpreference['party'].to_list())
interval_statistics = pd.DataFrame(columns=dfpreference['party'].to_list())
interval = pd.DataFrame(columns=['Pr[duel zisk > x %]'])
interval_now = pd.DataFrame(columns=['Pr[duel zisk > x %]'])
# for i in np.concatenate((np.arange(0, interval_max + 0.5, 0.5), np.array([2.55, 6.19, 10.97, 11.21, 16.13, 21.82, 24.91]))):
for i in np.concatenate((np.arange(interval_min, interval_max + 0.5, 0.5), np.array([]))):
    interval = interval.append({'Pr[duel zisk > x %]': i}, ignore_index=True)
    interval_now = interval_now.append({'Pr[duel zisk > x %]': i}, ignore_index=True)
    interval_statistics_aging = interval_statistics_aging.append((simulations_aging > (i / 100)).sum() / sample, ignore_index=True)
    interval_statistics = interval_statistics.append((simulations > (i / 100)).sum() / sample, ignore_index=True)

# write to GSheet
wsw = sh.worksheet('pravděpodobnosti_aktuální_aging')
interval[interval_statistics_aging.columns] = interval_statistics_aging
wsw.update('A1', [interval.columns.values.tolist()] + interval.values.tolist())

wsw = sh.worksheet('pravděpodobnosti_aktuální')
interval_now[interval_statistics.columns] = interval_statistics
wsw.update('A1', [interval_now.columns.values.tolist()] + interval_now.values.tolist())

# write datetime
wsw = sh.worksheet('preference, ze kterých se to počítá')
d = datetime.datetime.now().isoformat()
wsw.update('E2', d)

# save to history
# duels
history = pd.read_csv(path + 'history_2_duel_win.csv')
t = winning.reset_index().rename(columns={'index': 'duel'})
del t['p2']
t['datetime'] = d
t['date'] = today.isoformat()
t['g1'] = dfpreference['p']

pd.concat([history, t], ignore_index=True).to_csv(path + 'history_2_duel_win.csv', index=False)

# probability
history = pd.read_csv(path + 'history_2_prob.csv')
newly = pd.DataFrame(columns=history.columns)
cols = interval.columns
c0 = cols[0]
cols = cols.drop(cols[0])
for col in cols:
    t = interval[col].to_frame()
    t.columns = ['p1']
    t['less1'] = interval[c0]
    t['datetime'] = d
    t['g1'] = dfpreference[dfpreference['party'] == col]['gain1'].values[0]
    t['duel'] = col
    t['date'] = today.isoformat()
    newly = newly.append(t, ignore_index=True)

pd.concat([history, newly], ignore_index=True).to_csv(path + 'history_2_prob.csv', index=False)
