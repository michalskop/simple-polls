"""Add odds to GSheet."""

import datetime
import gspread
import pandas as pd

# source sheet
sheetkey = "1RGruTy8xbw5WOthXgUVZjwG_r4342ZCgZH-Q-7x4j6c"
path = "cz-eu-2024/"

# load data from GSheet
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)

# Mapping from our names
# tipsport
# mappingt = {

# }

# fortuna
mappingf = {
  "ANO": "ANO 2011",
  "SPOLU": "SPOLU",
  "SPD+Trikolóra": "SPD a Trikolora",
  "STAN": "Starostové a osobnosti pro Evropu",
  "Piráti": "Česká pirátská strana",
  "Stačilo!": "STAČILO!",
  "Přísaha": "PŘÍSAHA a MOTORISTÉ",
  "SOCDEM": "Sociální demokracie",
  "Svobodní": "Svobodní",
  "PRO": "PRO",
  "Zelení": "Zelení",
}
# candidates are keys
candidates = [k for k in mappingf.keys()]

# mappingn = {
#   "ANO": "ANO 2011",
#   "SPOLU": "SPOLU",
#   "SPD+Trikolóra": "SPD a Trikolora",
#   "STAN": "Starostové a osobnosti pro Evropu",
#   "Piráti": "Česká pirátská strana",
#   "Stačilo!": "STAČILO!",
#   "Přísaha": "PŘÍSAHA a MOTORISTÉ",
#   "SOCDEM": "Sociální demokracie",
#   "Svobodní": "Svobodní",
#   "PRO": "PRO",
#   "Zelení": "Zelení",
# }

# RANK
####################
ws = sh.worksheet('pořadí_aktuální_aging_cov')
dfr = pd.DataFrame(ws.get_all_records())

# # Tipsport
# urlt = "https://github.com/michalskop/tipsport.cz/raw/main/v3/data/5928059.csv"
# dft = pd.read_csv(urlt, encoding="utf-8")
# grows = {
#   'Ano': 30,
#   'Ne': 46
# }
# for s in ['Ano', 'Ne']:
#   rankt = []
#   rankt.append(dft[(dft['hypername'] == 'Nejvíc hlasů získá Ano/Ne') & (dft['name'] == s)].drop_duplicates(subset=['hypername', 'supername', 'name'], keep='last'))
#   rankt.append(dft[(dft['hypername'] == 'Umístění do 2. místa podle počtu hlasů Ano/Ne') & (dft['name'] == s)].drop_duplicates(subset=['hypername', 'supername', 'name'], keep='last'))
#   rankt.append(dft[(dft['hypername'] == 'Umístění do 3. místa podle počtu hlasů Ano/Ne') & (dft['name'] == s)].drop_duplicates(subset=['hypername', 'supername', 'name'], keep='last'))

#   rt = [[dft['date'].iloc[-1], s]]
#   for i, r in enumerate(rankt):
#     # break
#     rt.append([''])
#     for c in dfr.columns[1:]:
#       rt[i + 1].append(rankt[i][rankt[i]['supername'] == mappingt[c]].iloc[0]['odd'])
  
#   # add to GSheet
#   ws = sh.worksheet('pořadí_aktuální_aging_cov')
#   ws.update('A' + str(grows[s]), rt)

#   ws = sh.worksheet('pořadí_aktuální_aging')
#   ws.update('A' + str(grows[s]), rt)


# Fortuna
urlf = "https://github.com/michalskop/ifortuna.cz/raw/master/data/MCZ45367.csv"
dff = pd.read_csv(urlf, encoding="utf-8")
rankf = []
rankf.append(dff.drop_duplicates(subset=['event_name'], keep='last'))
urlf = "https://github.com/michalskop/ifortuna.cz/raw/master/data/MCZ45731.csv"
dff = pd.read_csv(urlf, encoding="utf-8")
rankf.append(dff.drop_duplicates(subset=['event_name'], keep='last'))
urlf = "https://github.com/michalskop/ifortuna.cz/raw/master/data/MCZ45732.csv"
dff = pd.read_csv(urlf, encoding="utf-8")
rankf.append(dff.drop_duplicates(subset=['event_name'], keep='last'))
urlf = "https://github.com/michalskop/ifortuna.cz/raw/master/data/MCZ45845.csv"
dff = pd.read_csv(urlf, encoding="utf-8")
rankf.append(dff.drop_duplicates(subset=['event_name'], keep='last'))
urlf = "https://github.com/michalskop/ifortuna.cz/raw/master/data/MCZ45846.csv"
dff = pd.read_csv(urlf, encoding="utf-8")
rankf.append(dff.drop_duplicates(subset=['event_name'], keep='last'))

growf = {
  'odds': 28,
  'odds2': 45
}
for s in ['odds', 'odds2']:

  rf = [[dff['date'].iloc[-1], s]]
  for i, r in enumerate(rankf):
    # break
    rf.append([''])
    for c in dfr.columns[1:]:
      if c in candidates:
        try:
          rf[i + 1].append(float(str(rankf[i][rankf[i]['event_name'] == mappingf[c]].iloc[0][s]).replace('\xa0','')))
        except:
          rf[i + 1].append('')

  ws = sh.worksheet('pořadí_aktuální_aging_cov')
  ws.update(range_name=('A' + str(growf[s])), values=rf)
  ws = sh.worksheet('pořadí_aktuální_aging')
  ws.update(range_name=('A' + str(growf[s])), values=rf)

# Nike
# not available
  

# DUELS
####################
# Tipsport  
# dfft = dft[dft['hypername'] == 'Kdo získá více hlasů'].drop_duplicates(subset=['hypername', 'supername', 'name'], keep='last')
# last_date = dft['date'].max()
# dfft = dfft[dfft['date'] == last_date]

# duelt = []
# for c1 in dfr.columns[1:]:
#   item = []
#   for c2 in dfr.columns[1:]:
#     exist = False
#     filtered = dfft[(dfft['supername'] == (mappingt[c1] + ' x ' + mappingt[c2]))]
#     if len(filtered) > 0:
#       filtered2 = filtered[(filtered['name'] == mappingt[c1])]
#       item.append(filtered2.iloc[0]['odd'])
#       exist = True
#     filtered = dfft[(dfft['supername'] == (mappingt[c2] + ' x ' + mappingt[c1]))]
#     if len(filtered) > 0:
#       filtered2 = filtered[(filtered['name'] == mappingt[c1])]
#       item.append(filtered2.iloc[0]['odd'])
#       exist = True
#     if not exist:
#       item.append('')
#   duelt.append(item)
# duelt.append([dfft['date'].iloc[-1]])

# ws = sh.worksheet('duely_aging')
# ws.update('B32', duelt)

# ws = sh.worksheet('duely_aging_cov')
# ws.update('B32', duelt)
  
# Fortuna
urlf2 = "https://github.com/michalskop/ifortuna.cz/raw/master/data/MCZ45733.csv"
dff2 = pd.read_csv(urlf2, encoding="utf-8")

dfff = dff2.drop_duplicates(subset=['event_name'], keep='last')
# remove old data
dfff = dfff[dfff['date'] == dfff['date'].iloc[-1]]

duelf = []
for c1 in dfr.columns[1:]:
  if c1 not in candidates:
    continue
  item = []
  for c2 in dfr.columns[1:]:
    if c2 not in candidates:
      continue
    filtered = dfff[(dfff['event_name'] == (mappingf[c1] + ' - ' + mappingf[c2]))]
    if len(filtered) > 0:
      item.append(filtered.iloc[0]['odds'])
    else:
      filtered = dfff[(dfff['event_name'] == (mappingf[c2] + ' - ' + mappingf[c1]))]
      if len(filtered) > 0:
        item.append(filtered.iloc[0]['odds2'])
      else:
        item.append('')
  duelf.append(item)
duelf.append([dfff['date'].iloc[-1]])

ws = sh.worksheet('duely_aging_cov')
ws.update(range_name='B34', values=duelf)

ws = sh.worksheet('duely_aging')
ws.update(range_name='B34', values=duelf)

# # Nike
# urln = "https://raw.githubusercontent.com/michalskop/nike.sk/main/v0/data/43804671.csv"
# dfn = pd.read_csv(urln, encoding="utf-8")

# dfnf = dfn[(dfn['header'] == 'Získa viac percent')].drop_duplicates(subset=['header', 'name', 'odds_name'], keep='last')

# dueln = []
# for c1 in dfr.columns[1:]:
#   item = []
#   for c2 in dfr.columns[1:]:
#     filtered = dfnf[(dfnf['name'] == (mappingn[c1] + ' - ' + mappingn[c2])) & (dfnf['odds_name'] == mappingn[c1])]
#     if len(filtered) > 0:
#       item.append(filtered.iloc[0]['odds'])
#     else:
#       filtered = dfnf[(dfnf['name'] == (mappingn[c2] + ' - ' + mappingn[c1])) & (dfnf['odds_name'] == mappingn[c1])]
#       if len(filtered) > 0:
#         item.append(filtered.iloc[0]['odds'])
#       else:
#         item.append('')
#   dueln.append(item)
# dueln.append([dfn['date'].iloc[-1]])

# ws = sh.worksheet('duely_aging_cov')
# ws.update('B62', dueln)

# ws = sh.worksheet('duely_aging')
# ws.update('B62', dueln)

# MORE THAN x%
####################
ws = sh.worksheet("pravděpodobnosti_aktuální_aging_cov")
dfmore = ws.col_values(1)[1:]
dfmore_end = dfmore.index('')
dfmore = dfmore[:dfmore_end]

# # Tipsport
# dftf = dft[dft['hypername'] == 'Počet hlasů v procentech'].drop_duplicates(subset=['hypername', 'supername', 'name'], keep='last')
# last_date = dftf['date'].max()
# dftf = dftf[dftf['date'] == last_date]
# gcols = {
#   '{}% a více': 'P',
#   'Méně než {}%': 'AC'
# }
# lastcol = {
#   'header1': 'AM',
#   'header2': 'Z'
# }
# # clear
# ws = sh.worksheet('pravděpodobnosti_aktuální_aging_cov')
# start_row = 134
# range_to_clear = f'{gcols["{}% a více"]}{start_row}:{lastcol["header2"]}{start_row + len(dfmore)}'
# ws.batch_clear([range_to_clear])
# range_to_clear = f'{gcols["Méně než {}%"]}{start_row}:{lastcol["header1"]}{start_row + len(dfmore)}'
# ws.batch_clear([range_to_clear])
# ws = sh.worksheet('pravděpodobnosti_aktuální_aging')
# range_to_clear = f'{gcols["{}% a více"]}{start_row}:{lastcol["header2"]}{start_row + len(dfmore)}'
# ws.batch_clear([range_to_clear])
# range_to_clear = f'{gcols["Méně než {}%"]}{start_row}:{lastcol["header1"]}{start_row + len(dfmore)}'
# ws.batch_clear([range_to_clear])

# for s in ['{}% a více', 'Méně než {}%']:
#   moret = []
#   for n in dfmore:
#     item = []
#     for c in dfr.columns[1:]:
#       filtered = dftf[(dftf['name'].eq(s.format(round(float(n) + 0.01, 2)))) & (dftf['supername'].eq(mappingt[c]))]
#       if len(filtered) > 0:
#         item.append(filtered.iloc[0]['odd'])
#       else:
#         item.append('')
#     moret.append(item)

#   ws = sh.worksheet('pravděpodobnosti_aktuální_aging_cov')
#   ws.update(gcols[s] + '134', moret)

#   ws = sh.worksheet('pravděpodobnosti_aktuální_aging')
#   ws.update(gcols[s] + '134', moret)


# Fortuna
# NOTE : not working now
# urlf2 = "https://raw.githubusercontent.com/michalskop/ifortuna.cz/master/data/MCZ45847.v2-1.csv"
# dff2 = pd.read_csv(urlf2, encoding="utf-8")

# dfff = dff2.drop_duplicates(subset=['event_name', 'event_link'], keep='last')
# last_date = dfff['date'].max()
# dfff = dfff[dfff['date'] == last_date]
# gcols = {
#   'header1': 'AC',
#   'header2': 'Q'
# }
# gstr = {
#   'header1': '- {}',
#   'header2': '+ {}'
# }
# lastcol = {
#   'header1': 'AM',
#   'header2': 'AA'
# }
# # clear
# # API limit: Max rows: 200, max columns: 12
# ws = sh.worksheet('pravděpodobnosti_aktuální_aging_cov')
# start_row = 141
# range_to_clear = f'{gcols["header2"]}{start_row}:{lastcol["header2"]}{start_row + len(dfmore)}'
# ws.batch_clear([range_to_clear])
# range_to_clear = f'{gcols["header1"]}{start_row}:{lastcol["header1"]}{start_row + len(dfmore)}'
# ws.batch_clear([range_to_clear])
# ws = sh.worksheet('pravděpodobnosti_aktuální_aging')
# range_to_clear = f'{gcols["header2"]}{start_row}:{lastcol["header2"]}{start_row + len(dfmore)}'
# ws.batch_clear([range_to_clear])
# range_to_clear = f'{gcols["header1"]}{start_row}:{lastcol["header1"]}{start_row + len(dfmore)}'
# ws.batch_clear([range_to_clear])

# i = 1
# for s in ['header1', 'header2']:
#   moref = []
#   for n in dfmore:
#     item = []
#     for c in dfr.columns[1:]:
#       if c not in candidates:
#         continue
#       filtered = dfff[(dfff['event_name'] == mappingf[c]) & (dfff[s].eq(gstr[s].format(float(n) + 0.01)))]
#       if len(filtered) > 0:
#         item.append(filtered.iloc[0]['odd' + str(i)])
#       else:
#         item.append('')
#     moref.append(item)
#   i += 1
#   ws = sh.worksheet('pravděpodobnosti_aktuální_aging_cov')
#   ws.update(gcols[s] + str(start_row), moref)
#   ws = sh.worksheet('pravděpodobnosti_aktuální_aging')
#   ws.update(gcols[s] + str(start_row), moref)

# # Nike
# gcols = {
#   'viac ako': 'P',
#   'menej ako': 'AC'
# }
# lastcol = {
#   'header1': 'Z',
#   'header2': 'AM'
# }
# dfnf = dfn[(dfn['header'] == 'Počet percent')].drop_duplicates(subset=['header', 'name', 'odds_name'], keep='last')
# last_date = dfnf['date'].max()
# dfnf = dfnf[dfnf['date'] == last_date]

# # clear
# # API limit: Max rows: 200, max columns: 12
# ws = sh.worksheet('pravděpodobnosti_aktuální_aging_cov')
# start_row = 308
# range_to_clear = f'{gcols["viac ako"]}{start_row}:{lastcol["header2"]}{start_row + len(dfmore)}'
# ws.batch_clear([range_to_clear])
# range_to_clear = f'{gcols["menej ako"]}{start_row}:{lastcol["header1"]}{start_row + len(dfmore)}'
# ws.batch_clear([range_to_clear])
# ws = sh.worksheet('pravděpodobnosti_aktuální_aging')
# range_to_clear = f'{gcols["viac ako"]}{start_row}:{lastcol["header2"]}{start_row + len(dfmore)}'
# ws.batch_clear([range_to_clear])
# range_to_clear = f'{gcols["menej ako"]}{start_row}:{lastcol["header1"]}{start_row + len(dfmore)}'
# ws.batch_clear([range_to_clear])


# for s in ['viac ako', 'menej ako']:
#   moren = []
#   for n in dfmore:
#     item = []
#     for c in dfr.columns[1:]:
#       # break
#       filtered = dfnf[(dfnf['name'] == mappingn[c]) & (dfnf['odds_name'].eq(s + ' ' + '{:.2f}'.format(float(n))))]
#       if len(filtered) > 0:
#         item.append(filtered.iloc[0]['odds'])
#       else:
#         item.append('')

#     moren.append(item)

#   ws = sh.worksheet('pravděpodobnosti_aktuální_aging_cov')
#   ws.update(gcols[s] + '398', moren)
#   ws = sh.worksheet('pravděpodobnosti_aktuální_aging')
#   ws.update(gcols[s] + '398', moren)