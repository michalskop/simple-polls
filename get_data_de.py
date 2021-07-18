"""Process data from europeelects.eu."""

import datetime
import numpy as np
import pandas as pd
import gspread
# import mygsheet

url = "https://filipvanlaenen.github.io/eopaod/de-N.csv"

raw = pd.read_csv(url)

# filter only current term
raw = raw[raw['Fieldwork End'] > '2017-09-24']

not_parties = ['Polling Firm', 'Commissioners', 'Fieldwork Start',
       'Fieldwork End', 'Scope', 'Sample Size',
       'Sample Size Qualification', 'Participation', 'Precision', 'Other']

others = 'Other'

parties = []
for c in raw.columns.values:
    if c not in not_parties:
        parties.append(c)

parties_all = parties + [others]
 

# Gsheet
gc = gspread.service_account()
# sh0 = gc.open_by_key("1a9zd3ThneSR7JN7-wj4uw5hBBggY9NyguFYIb4jbAc0")
sh = gc.open_by_key("1O2j5aDkUVZawVgnQ9Flt8e-SuCZJxsulpcGHBKGDb_k")

# Copy template
# mygsheet.copy_teplate(sh0, sh)

# pollsters
ws = sh.worksheet('pollsters')
columns = ws.row_values(1)
pollsters = raw['Polling Firm'].unique()

df = pd.DataFrame(columns=ws.row_values(1))
for p in pollsters:
    item = {
        'id': p,
        'name': p,
        'abbreviation': p,
        'done_by': 'makabot - Michal Škop'
    }
    df = df.append(item, ignore_index=True)
df = df.fillna('')

ws.update([df.columns.values.tolist()] + df.values.tolist())

# topics
ws = sh.worksheet('topics')
columns = ws.row_values(1)
df = pd.DataFrame(columns=ws.row_values(1))

items = [
    {
        'id': 'model Germany',
        'name': 'Model pro volby - Germany',
        'region': 'de',
        'region_name': 'Germany',
        'done_by': 'makabot - Michal Škop'
    }
]
for item in items:
    df = df.append(item, ignore_index=True)

df = df.fillna('')

ws.update([df.columns.values.tolist()] + df.values.tolist())

# polls
ws = sh.worksheet('polls')
columns = ws.row_values(1)
df = pd.DataFrame(columns=ws.row_values(1))
day0 = '1899-12-30'

# raw.shape[0]

df['identifier'] = raw.shape[0] - raw.reset_index()['index']
df['pollster:id'] = raw['Polling Firm']
df['start_date'] = raw['Fieldwork Start']
df['end_date'] = raw['Fieldwork End']
df['sponsor'] = raw['Commissioners']
df['n'] = raw['Sample Size']
df['done_by'] = 'Makabot - Michal Škop'

df = df.sort_values('identifier')

df = df.fillna('')

df['start_date'] = df['start_date'].apply(lambda x: (datetime.datetime.fromisoformat(x) - datetime.datetime.fromisoformat(day0)).days)
df['end_date'] = df['end_date'].apply(lambda x: (datetime.datetime.fromisoformat(x) - datetime.datetime.fromisoformat(day0)).days)

ws.update([df.columns.values.tolist()] + df.values.tolist())

ws.format('D2:E' + str(raw.shape[0] + 1), {"numberFormat": {"type": "DATE", "pattern": "YYYY-MM-DD"}})

# add average
arrn = np.arange(2, raw.shape[0] + 2)
arr = [(lambda x: ['=round(average(D' + str(x) + ',E' + str(x) + "))"])(x) for x in arrn]
ws.update('P2', arr, raw=False)
ws.format('P2:P' + str(raw.shape[0] + 1), {"numberFormat": {"type": "DATE", "pattern": "YYYY-MM-DD"}})
# add days to elections
election_day = '2021-09-26'
election_value = (datetime.datetime.fromisoformat(election_day) - datetime.datetime.fromisoformat(day0)).days
arr = [(lambda x: ['=' + str(election_value) + '-P' + str(x)])(x) for x in arrn]
ws.update('Q2', arr, raw=False)
ws.format('Q2:Q' + str(raw.shape[0] + 1), {"numberFormat": {"type": "NUMBER", "pattern": "#,##0"}})

# questions
ws = sh.worksheet('questions')
columns = ws.row_values(1)
df = pd.DataFrame(columns=ws.row_values(1))

def _precision2identifier(p):
    if p == 'S%':
        return 'mandáty'
    else:
        return 'model'

def _precision2topic(p):
    if p == 'S%':
        return 'mandáty Germany'
    else:
        return 'model Germany'

df['poll:identifier'] = raw.shape[0] - raw.reset_index()['index']
df['identifier'] = raw['Precision'].apply(_precision2identifier)
df['pollster:id'] = raw['Polling Firm']
df['topic:id'] = raw['Precision'].apply(_precision2topic)
df['done_by'] = 'Makabot - Michal Škop'

df = df.sort_values('poll:identifier')

df = df.fillna('')
ws.update([df.columns.values.tolist()] + df.values.tolist())

# choices
ws = sh.worksheet('choices')
columns = ws.row_values(1)
df = pd.DataFrame(columns=ws.row_values(1))

df['id'] = parties + ['Other']
df['name'] = df['id']
df['abbreviation'] = df['id']
df['done_by'] = 'Makabot - Michal Škop'

df = df.fillna('')
ws.update([df.columns.values.tolist()] + df.values.tolist())

# data
ws = sh.worksheet('data')
columns = ws.row_values(1)
df = pd.DataFrame(columns=ws.row_values(1))

def _valueprecision2value(v, p):
    if v == 'Not Available':
        return ''
    if p == 'S%':
        return v.replace('%', '')
    else:
        return v

data = []
for index, row in raw.iterrows():
    for p in parties_all:
        item = {
            "question:identifier": _precision2identifier(row['Precision']),
            "poll:identifier": raw.shape[0] - index,
            "pollster:id": row['Polling Firm'],
            "choice_id": p,
            "topic_id":  _precision2topic(row['Precision']),
            "value": _valueprecision2value(row[p], row['Precision']),
            "done_by": "Makabot - Michal Škop",
            "notes": ""
        }
        data.append(item)

df = pd.DataFrame(data)
df = df.sort_values('poll:identifier')

df = df.fillna('')
ws.update([df.columns.values.tolist()] + df.values.tolist(), value_input_option='USER_ENTERED')
