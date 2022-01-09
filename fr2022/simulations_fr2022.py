"""Simulations for FR 2022."""

import datetime
import gspread
import math
import numpy as np
import pandas as pd
import scipy.stats
# from matplotlib import pyplot as plt

election_date = '2022-04-10'
election_day = datetime.date.fromisoformat(election_date)
today = datetime.date.today()   # it changes later !!!
sample_n = 1000 # used in statistical error
re_coef = 0.6 # random error coefficient
sample = 2000 # number of simulation
interval_max = 40 # highest gain to calc probability
# source sheet
sheetkey = "1xGJfKlN1UwzoMI71-UMAwlZMDQrKyx7aBgV5_Fn6x78"

# load data from GSheet
gc = gspread.service_account()

sh = gc.open_by_key(sheetkey)

ws = sh.worksheet('preference, ze kterých se to počítá')
dfpreference = pd.DataFrame(ws.get_all_records())
dfpreference['p'] = dfpreference['gain'] / 100
# today
today = datetime.date.fromisoformat(dfpreference['date'][0])

# aging curve 
def aging_coeff(day1, day2):
    diff = abs((day2 - day1).days)
    return pow(diff, 1.15) / diff

# p = dfpreference
# n = sample_n
# normal error
def normal_error(p, n, coef = 1):
    p['sdx'] = (n * p['p'] * (1 - p['p'])).apply(math.sqrt) / n * coef
    p['normal_error'] = scipy.stats.norm.rvs(loc=0, scale=p['sdx'])
    return p

# uniform_error as function of normal error
def uniform_error(p, n, coef = 1):
    p['sdx'] = (n * p['p'] * (1 - p['p'])).apply(math.sqrt) / n * coef
    p['uniform_error'] = scipy.stats.uniform.rvs(loc=(-1 * p['sdx'] * math.sqrt(3)), scale=(2 * p['sdx'] * math.sqrt(3)))
    return p

# simulations
simulations = pd.DataFrame(columns=dfpreference['party'].to_list())
simulations_aging = pd.DataFrame(columns=dfpreference['party'].to_list())
aging = aging_coeff(today, election_day)
for i in range(0, sample):
    p = normal_error(dfpreference, sample_n)
    p = uniform_error(p, sample_n, 1.5 * 0.9)
    p['estimate'] = p['normal_error'] + p['uniform_error'] + p['p']
    p['estimate_aging'] = aging * (p['normal_error'] + p['uniform_error']) + p['p']
    simx = dict(zip(dfpreference['party'].to_list(), p['estimate']))
    simxa = dict(zip(dfpreference['party'].to_list(), p['estimate_aging']))
    simulations = simulations.append(simx, ignore_index=True)
    simulations_aging = simulations_aging.append(simxa, ignore_index=True)

# rank matrix (somehow did not work directly)
ranks = simulations.loc[0:sample,:].rank(axis=1, ascending=False)
ranks_statistics = pd.DataFrame(index=ranks.columns)
ranks_aging = simulations_aging.loc[0:sample,:].rank(axis=1, ascending=False)
ranks_statistics_aging = pd.DataFrame(index=ranks_aging.columns)
for i in range(1, len(ranks.columns)):
    ranks_statistics[str(i)] = pd.DataFrame((ranks <= i).sum() / sample).rename(columns={0: str(i)})
    ranks_statistics_aging[str(i)] = pd.DataFrame((ranks_aging <= i).sum() / sample).rename(columns={0: str(i)})

# less than
interval_statistics = pd.DataFrame(columns=dfpreference['party'].to_list())
interval_statistics_aging = pd.DataFrame(columns=dfpreference['party'].to_list())
# for i in np.concatenate((np.arange(0, interval_max + 0.5, 0.5), np.array([2.55, 6.19, 10.97, 11.21, 16.13, 21.82, 24.91]))):
for i in np.concatenate((np.arange(0, interval_max + 0.5, 0.5), np.array([]))):    
    interval_statistics = interval_statistics.append((simulations > (i / 100)).sum() / sample, ignore_index=True)
    interval_statistics_aging = interval_statistics_aging.append((simulations_aging > (i / 100)).sum() / sample, ignore_index=True)

# duels
duels = pd.DataFrame(columns = ranks.columns, index=ranks.columns)
for i in ranks.columns:
    for j in ranks.columns:
        p = (sum(ranks[i] >= ranks[j])) / sample
        duels[i][j] = p
duels_aging = pd.DataFrame(columns = ranks_aging.columns, index=ranks_aging.columns)
for i in ranks_aging.columns:
    for j in ranks_aging.columns:
        p = (sum(ranks_aging[i] >= ranks_aging[j])) / sample
        duels_aging[i][j] = p

# history
# ranks_statistics_hist = ranks_statistics.stack().reset_index()
# ranks_statistics_hist['now'] = datetime.datetime.now().isoformat()

# ranks_statistics_aging_hist = ranks_statistics_aging.stack().reset_index()
# ranks_statistics_aging_hist['now'] = datetime.datetime.now().isoformat()

# interval_statistics_hist = interval_statistics.stack().reset_index()
# interval_statistics_hist['now'] = datetime.datetime.now().isoformat()

# interval_statistics_aging_hist = interval_statistics_aging.stack().reset_index()
# interval_statistics_aging_hist['now'] = datetime.datetime.now().isoformat()

# WRITE TO SHEET
wsw = sh.worksheet('pořadí_aktuální')
wsw.update('B1', [ranks_statistics.transpose().columns.values.tolist()] + ranks_statistics.transpose().values.tolist())

wsw = sh.worksheet('pořadí_aktuální_aging')
wsw.update('B1', [ranks_statistics_aging.transpose().columns.values.tolist()] + ranks_statistics_aging.transpose().values.tolist())

wsw = sh.worksheet('pravděpodobnosti_aktuální')
wsw.update('B1', [interval_statistics.columns.values.tolist()] + interval_statistics.values.tolist())

wsw = sh.worksheet('pravděpodobnosti_aktuální_aging')
wsw.update('B1', [interval_statistics_aging.columns.values.tolist()] + interval_statistics_aging.values.tolist())

# wsw = sh.worksheet('pořadí_historie')
# existing = len(wsw.get_all_values())
# wsw.update('A' + str(existing + 1), ranks_statistics_hist.values.tolist())

# wsw = sh.worksheet('pořadí_historie_aging')
# existing = len(wsw.get_all_values())
# wsw.update('A' + str(existing + 1), ranks_statistics_aging_hist.values.tolist())

# wsw = sh.worksheet('pravděpodobnosti_historie')
# existing = len(wsw.get_all_values())
# wsw.update('A' + str(existing + 1), interval_statistics_hist.values.tolist())

# wsw = sh.worksheet('pravděpodobnosti_historie_aging')
# existing = len(wsw.get_all_values())
# wsw.update('A' + str(existing + 1), interval_statistics_aging_hist.values.tolist())

wsw = sh.worksheet('duely')
wsw.update('B2', [duels.columns.values.tolist()] + duels.values.tolist())

wsw = sh.worksheet('duely_aging')
wsw.update('B2', [duels_aging.columns.values.tolist()] + duels_aging.values.tolist())

wsw = sh.worksheet('preference, ze kterých se to počítá')
d = datetime.datetime.now().isoformat()
wsw.update('D2', d)

# TESTS

# ranks_statistics_hist = ranks_statistics.stack().reset_index()
# ranks_statistics_hist['now'] = datetime.datetime.now().isoformat()

# normal_error(dfpreference, sample_n)['value']
# uniform_error(dfpreference, sample_n)

# a = [2,4,6]   
# a.append(scipy.stats.uniform.rvs(loc=(-1 * math.sqrt(3)), scale=(2 * 1 * math.sqrt(3))))
# mean = sum(a) / len(a)
# variance = sum([((x - mean) ** 2) for x in a]) / len(a)
# sd = variance ** 0.5
# print(sd)

# a = []
# for i in range(0, 10000):
#     uni = scipy.stats.uniform.rvs(loc=(-1 * math.sqrt(3)), scale=(2 * 1 * math.sqrt(3)))
#     norm = scipy.stats.norm.rvs(loc=0, scale=1)
#     a.append(uni)
    
# plt.hist(a)
# mean = sum(a) / len(a)
# variance = sum([((x - mean) ** 2) for x in a]) / len(a)
# sd = variance ** 0.5
# print(sd)

# s = sum(p['rand'])
# ss = sum(p['p'])
# p['value'] = p['rand'] / s * ss

# std = (sample_n * dfpreference['p'] * (1 - dfpreference['p'])).apply(math.sqrt) / sample_n
# scipy.stats.norm.rvs(loc=0, scale=std)


# aging_coeff(election_day, today)