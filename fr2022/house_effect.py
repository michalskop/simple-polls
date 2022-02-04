"""House effect."""

import datetime
import gspread
import numpy as np
import pandas as pd

sheetkey = "1JxLDU_mYDlcRbh45BA9wLByyPYMmGCPDbYXtRlPve8w"
csv_file = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTgg1x2P_SS-52MmixvFPtK1JdiEQOeWcvut1Xk65JJFq6KGs0sUxcNADY8LKgc_DK3PEd07S2piyri/pub?gid=0&single=true&output=csv"
path = 'fr2022/'

polls_min = 4

selected = ['Macron', 'Le Pen', 'Pécresse' ,'Zemmour', 'Mélenchon']


# GSheet
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)

# read data
data = pd.read_csv(csv_file)

# pollsters
pollsters = data['pollster:id'].unique().tolist()
pollsters.remove('volby')

# pollster with mininal polls
pollsters_sorted = data.groupby('pollster:id').size().sort_values(ascending=False)
pollsters_selected = pollsters_sorted[pollsters_sorted >= polls_min].index.tolist()

# dates
selected_dates = data['middle_date'].unique().tolist()
selected_dates.pop(0)   # volby

# prepare empty dataframes
ma = {}
for s in selected:
    ma[s] = pd.DataFrame(columns=['middle_date', 'middle_day'] + pollsters_selected)

# calculate partial moving average
data['middle_day'] = data['middle_date'].apply(lambda x: datetime.datetime.fromisoformat(x))

for date in selected_dates:
    d = datetime.datetime.fromisoformat(date)
    for s in selected:
        row = {'middle_date': date, 'middle_day': d}
        for p in pollsters_selected:
            t = data[data['pollster:id'] == p]
            t['weight'] = np.nan
            t['weight'] = (1 / 2) ** (abs(t['middle_day'] - d).apply(datetime.timedelta.total_seconds) / 60 / 60 / 24 / 15)
            v = (t[s] * t['weight']).sum() / t['weight'].sum()
            row[p] = v
        ma[s] = ma[s].append(row, ignore_index=True)

# calculate overall moving averages
moving_averages = pd.DataFrame(columns=['middle_date', 'middle_day'] + selected)
for date in selected_dates:
    d = datetime.datetime.fromisoformat(date)
    row = {'middle_date': date, 'middle_day': d}
    for s in selected:
        row[s] = ma[s][ma[s]['middle_date'] == date][pollsters_selected].mean(axis=1).values[0]
    moving_averages = moving_averages.append(row, ignore_index=True)

# house effect
house_effect = pd.DataFrame(columns = ['pollster:id', 'choice:id', 'mean', 'sd', 'n'])
for p in pollsters_selected:
    pollstert = data[data['pollster:id'] == p][['middle_date'] + selected]
    ptmerged = pollstert.merge(moving_averages, on='middle_date', how='left')
    for s in selected:
        mean = np.mean((ptmerged[s + '_x'] - ptmerged[s + '_y']))
        sd = np.std((ptmerged[s + '_x'] - ptmerged[s + '_y']))
        house_effect = house_effect.append({
            'choice:id': s,
            'pollster:id': p,
            'mean': mean,
            'sd': sd,
            'n': len(ptmerged)
        }, ignore_index=True)

# write to Gsheet
he = house_effect.pivot_table(values=['mean', 'sd'], index='choice:id', columns='pollster:id').reset_index()

wsw = sh.worksheet('house effect')
wsw.update('A2', he['choice:id'].apply(lambda x: [x]).to_list())
wsw.update('B1', [he['mean'].columns.values.tolist()] + he['mean'].values.tolist())
wsw.update(chr(ord('B') + 2 + len(he['mean'].columns.values.tolist())) + "1", [he['sd'].columns.values.tolist()] + he['sd'].values.tolist())

# write to file
he.to_csv(path + 'house_effect.csv', index=False)
