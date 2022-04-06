"""Get poll data."""

# source: https://github.com/nsppolls/nsppolls

import datetime
import gspread
import json
import requests

url = "https://raw.githubusercontent.com/nsppolls/nsppolls/master/presidentielle.json"

key = "1JxLDU_mYDlcRbh45BA9wLByyPYMmGCPDbYXtRlPve8w"

# first_date = '2021-12-01'
first_date = '2000-12-01'
now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

r = requests.get(url)
data = json.loads(r.text)

# open google sheer
gc = gspread.service_account()
sh = gc.open_by_key(key)

# first round
worksheet_list = sh.worksheets()
for ws in worksheet_list:
    if ws.title == "Premier tour":
        worksheet = ws
values_list = worksheet.row_values(1)
i = 0
for value in values_list:
    if value == "start_date":
        date_index = i
    i += 1
values_list = worksheet.row_values(2)
election_date = datetime.datetime.fromisoformat(values_list[date_index]).date()

header = ['pollster:id', 'start_date', 'end_date', 'n', 'middle_date', 'days to elections', 'last_updated']
polls = []
for row in data:
    if row['debut_enquete'] > first_date:
        start_date = datetime.datetime.fromisoformat(row['debut_enquete']).date()
        end_date = datetime.datetime.fromisoformat(row['fin_enquete']).date()
        item = {
            'pollster:id': row['nom_institut'],
            'start_date': row['debut_enquete'],
            'end_date': row['fin_enquete'],
            'n': row['echantillon'],
            'middle_date': (start_date + (end_date - start_date) / 2).isoformat(),
            'days to elections': (election_date - end_date).days,
            'last_updated': None
        }
        for tour in row['tours']:
            if tour['tour'] == "Premier tour":
                i = 0
                for hypothesis in tour['hypotheses']:
                    if i == 0:
                    # if hypothesis['hypothese'] is None:
                        item['candidates'] = hypothesis['candidats']
                        polls.append(item)
                    else:
                        print('hypothese: ', hypothesis['hypothese'])
                    i += 1

# get candidates
polls = sorted(polls, key=lambda x: x['middle_date'], reverse=True)
candidate_names = []
candidate_sorted = {}
for poll in polls:
    for name in poll['candidates']:
        candidate_name = ' '.join(name['candidat'].split(' ')[1:])
        if candidate_name not in candidate_names:
            candidate_names.append(candidate_name)
            candidate_sorted[candidate_name] = name['intentions']

candidates_sorted = sorted(candidate_sorted, key=candidate_sorted.get, reverse=True)

# create rows
rows = []
for poll in polls:
    row = []
    for h in header:
        row.append(poll[h])
    row.append(None)
    ln = len(row)
    t = [0] * len(candidates_sorted)
    for c in poll['candidates']:
        name = ' '.join(c['candidat'].split(' ')[1:])
        index = candidates_sorted.index(name)
        t[index] = c['intentions']
    row = row + t
    rows.append(row)

# update sheet
uheader = header + [None] + candidates_sorted
worksheet.update('A1', [uheader])
worksheet.update_cell(2, header.index('last_updated') + 1, now)
worksheet.update('A3', rows)

# SECOND ROUND
# create all worksheets
worksheet_names = []
for ws in worksheet_list:
    if ws.title != "Premier tour":
        worksheet_names.append(ws.title)

election_date = '2022-04-24'
election_row = ['volby', election_date, election_date, None, election_date, None]
election_day = datetime.datetime.fromisoformat(election_date).date()

for row in data:
    if row['debut_enquete'] > first_date:
        for tour in row['tours']:
            if tour['tour'] == "Deuxième tour":
                for hypothesis in tour['hypotheses']:
                    candidate_names = []
                    for name in hypothesis['candidats']:
                        candidate_names.append(' '.join(name['candidat'].split(' ')[1:]))
                    candidate_names = sorted(candidate_names, key=lambda x: x[0])
                    ws_title = " / ".join(candidate_names)
                    if ws_title not in worksheet_names:
                        worksheet = sh.add_worksheet(title=ws_title, rows=200, cols=len(header) + len(candidate_names) + 1)
                        worksheet_names.append(ws_title)

                        worksheet.update('A1', [header + [None] + candidate_names])
                        worksheet.update('A2', [election_row])
                        worksheet.freeze(rows=2)

# add data to all worksheets
for ws_name in worksheet_names:
    polls = []
    for row in data:
        if row['debut_enquete'] > first_date:
            for tour in row['tours']:
                if tour['tour'] == "Deuxième tour":
                    for hypothesis in tour['hypotheses']:
                        candidate_names = []
                        for name in hypothesis['candidats']:
                            candidate_names.append(' '.join(name['candidat'].split(' ')[1:]))
                        candidate_names = sorted(candidate_names, key=lambda x: x[0])
                        if ws_name == " / ".join(candidate_names):
                            start_date = datetime.datetime.fromisoformat(row['debut_enquete']).date()
                            end_date = datetime.datetime.fromisoformat(row['fin_enquete']).date()
                            item = {
                                'pollster:id': row['nom_institut'],
                                'start_date': row['debut_enquete'],
                                'end_date': row['fin_enquete'],
                                'n': row['echantillon'],
                                'middle_date': (start_date + (end_date - start_date) / 2).isoformat(),
                                'days to elections': (election_day - end_date).days,
                                'last_updated': None
                            }
                            for c in hypothesis['candidats']:
                                item[" ".join(c['candidat'].split(' ')[1:])] = c['intentions']
                            polls.append(item)
                            polls = sorted(polls, key=lambda x: x['middle_date'], reverse=True)
    
    rows = []
    candidate_names = ws_name.split(' / ')
    for poll in polls:
        row = []
        for h in header:
            row.append(poll[h])
        row.append(None)
        ln = len(row)
        t = [0] * len(candidate_names)
        for c in poll:
            if c in candidate_names:
                index = candidate_names.index(c)
                t[index] = poll[c]
        row = row + t
        rows.append(row)

    worksheet = sh.worksheet(ws_name)
    worksheet.update('A3', rows)
    worksheet.update_cell(2, header.index('last_updated') + 1, now)
