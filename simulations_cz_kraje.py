"""Simulations for CZ regions."""

import datetime
import gspread
import math
import numpy as np
import pandas as pd
import scipy.stats

election_date = '2021-10-08'
election_day = datetime.date.fromisoformat(election_date)
today = datetime.date.today()   # it changes later !!!
sample_n = 1000 # used in statistical error
re_coefs = [1.5, 2] # random error coefficient
sample = 100 # number of simulation
# interval_max = 40 # highest gain to calc probability
# source sheet
sheetkey = "1gJ0fv56qEUPSX3yUwn_F0qszP6jDmd6C_xzTgGWaqIU"

lastresultsfile = "kraje_weights.csv"


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

# last results
# there are some new parties with approx. votes (Přísaha), only the 'rates' are ok
# last_results = pd.read_csv(lastresultsfile, encoding='utf-8')
# lrpt = pd.pivot_table(last_results, index='party_code', columns='region_code', values='votes')
# lrpts = lrpt.sum(axis=1)
# last_rates = lrpts / lrpts.sum()
# regional_results_r = lrpt / lrpt.sum()
# rates = (regional_results_r.T / last_rates)
rates_raw = pd.read_csv(lastresultsfile)
rates = pd.pivot_table(rates_raw, columns='party', index='region', values='W Average', aggfunc=np.sum)

# simulations
j = 0
# region = 'Plzeňský kraj'
for re_coef in re_coefs:
    wsw = sh.worksheet('kraje_aktuální')
    wsw.update('A' + str(j * 10 + 1), re_coef)
    wsw = sh.worksheet('kraje_aging')
    wsw.update('A' + str(j * 10 + 1), re_coef)
    for region in rates.index:
        print(region)
        df_regional_preferences = rates.T[region].to_frame().reset_index().merge(dfpreference, how='left', left_on='party', right_on='party').rename(columns={'p': 'p0'})

        df_regional_preferences['p'] = df_regional_preferences['p0'] * df_regional_preferences[region]

        simulations = pd.DataFrame(columns=df_regional_preferences['party'].to_list())
        simulations_aging = pd.DataFrame(columns=df_regional_preferences['party'].to_list())
        aging = aging_coeff(today, election_day)
        for i in range(0, sample):
            p = normal_error(df_regional_preferences, sample_n)
            p = uniform_error(p, sample_n, 1.5 * 0.9)
            p['estimate'] = re_coef * (p['normal_error'] + p['uniform_error']) + p['p']
            p['estimate_aging'] = re_coef * aging * (p['normal_error'] + p['uniform_error']) + p['p']
            simx = dict(zip(df_regional_preferences['party'].to_list(), p['estimate']))
            simxa = dict(zip(df_regional_preferences['party'].to_list(), p['estimate_aging']))
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

        # WRITE TO SHEET
        wsw = sh.worksheet('kraje_aktuální')
        wsw.update('B' + str(j * 10 + 2), region)
        wsw.update('C' + str(j * 10 + 2), [ranks_statistics.transpose().columns.values.tolist()] + ranks_statistics.transpose().values.tolist())

        wsw = sh.worksheet('kraje_aging')
        wsw.update('B' + str(j * 10 + 2), region)
        wsw.update('C' + str(j * 10 + 2), [ranks_statistics_aging.transpose().columns.values.tolist()] + ranks_statistics_aging.transpose().values.tolist())

        j += 1