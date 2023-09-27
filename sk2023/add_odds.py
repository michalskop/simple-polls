"""Add odds to GSheet."""

import datetime
import gspread
import pandas as pd

sheetkey = "1bRgInxsjoW4N9LVu6Tu79T8bYGIXPg330HGATj9cO9c"
path = "sk2023/"

# load data from GSheet
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)

# Mapping from our names
mappingt = {
  'SMER-SD': 'SMER-SD',
  'PS': 'Progresívne Slovensko',
  'HLAS-SD': 'HLAS-SD',
  'Republika': 'Republika',
  'SME Rodina': 'Sme rodina',
  'SaS': 'SaS',
  'KDH': 'KDH',
  'OĽaNO': 'OĽANO A PRIATELIA',
  'SNS': 'SNS',
  'Demokrati': 'Demokrati',
  'Aliancia': 'Aliancia',
  'Kotleba-ĽSNS': 'ĽS Naše Slovensko'
}

mappingf = {
  'SMER-SD': 'Smer',
  'PS': 'Progresívne Slovensko',
  'HLAS-SD': 'Hlas',
  'Republika': 'Republika',
  'SME Rodina': 'Sme rodina',
  'SaS': 'SaS',
  'KDH': 'KDH',
  'OĽaNO': 'OĽANO A PRIATELIA',
  'SNS': 'SNS',
  'Demokrati': 'Demokrati',
  'Aliancia': 'Aliancia',
  'Kotleba-ĽSNS': 'ĽS Naše Slovensko'
}

mappingn = {
  'SMER-SD': 'Smer',
  'PS': 'PS',
  'HLAS-SD': 'Hlas',
  'Republika': 'Republika',
  'SME Rodina': 'Sme rodina',
  'SaS': 'SaS',
  'KDH': 'KDH',
  'OĽaNO': 'OĽaNO',
  'SNS': 'SNS',
  'Demokrati': 'Demokrati',
  'Aliancia': 'Aliancia',
  'Kotleba-ĽSNS': 'ĽSNS'
}

# RANK
####################
ws = sh.worksheet('pořadí_aktuální_aging_cov')
dfr = pd.DataFrame(ws.get_all_records())

# Tipsport
urlt = "https://github.com/michalskop/tipsport.cz/raw/main/v3/data/5209621.csv"
dft = pd.read_csv(urlt, encoding="utf-8")
grows = {
  'Ano': 37,
  'Ne': 55
}
for s in ['Ano', 'Ne']:
  rankt = []
  rankt.append(dft[(dft['hypername'] == 'Nejvíc hlasů získá Ano/Ne') & (dft['name'] == s)].drop_duplicates(subset=['hypername', 'supername', 'name'], keep='last'))
  rankt.append(dft[(dft['hypername'] == 'Umístění do 2. místa podle počtu hlasů Ano/Ne') & (dft['name'] == s)].drop_duplicates(subset=['hypername', 'supername', 'name'], keep='last'))
  rankt.append(dft[(dft['hypername'] == 'Umístění do 3. místa podle počtu hlasů Ano/Ne') & (dft['name'] == s)].
  drop_duplicates(subset=['hypername', 'supername', 'name'], keep='last'))
  rankt.append(dft[(dft['hypername'] == 'Umístění do 4. místa podle počtu hlasů Ano/Ne') & (dft['name'] == s)].drop_duplicates(subset=['hypername', 'supername', 'name'], keep='last'))
  rankt.append(dft[(dft['hypername'] == 'Umístění do 5. místa podle počtu hlasů Ano/Ne') & (dft['name'] == s)].drop_duplicates(subset=['hypername', 'supername', 'name'], keep='last'))
  rankt.append(dft[(dft['hypername'] == 'Umístění do 6. místa podle počtu hlasů Ano/Ne') & (dft['name'] == s)].drop_duplicates(subset=['hypername', 'supername', 'name'], keep='last'))
  rankt.append(dft[(dft['hypername'] == 'Umístění do 7. místa podle počtu hlasů Ano/Ne') & (dft['name'] == s)].drop_duplicates(subset=['hypername', 'supername', 'name'], keep='last'))

  rt = [[dft['date'].iloc[-1], s]]
  for i, r in enumerate(rankt):
    # break
    rt.append([''])
    for c in dfr.columns[1:]:
      rt[i + 1].append(rankt[i][rankt[i]['supername'] == mappingt[c]].iloc[0]['odd'])

  ws = sh.worksheet('pořadí_aktuální_aging_cov')
  ws.update('A' + str(grows[s]), rt)

  ws = sh.worksheet('pořadí_aktuální_aging')
  ws.update('A' + str(grows[s]), rt)


# Fortuna
urlf = "https://github.com/michalskop/ifortuna.cz/raw/master/data/MCZ25950.csv"
dff = pd.read_csv(urlf, encoding="utf-8")
rankf = []
rankf.append(dff.drop_duplicates(subset=['event_name'], keep='last'))
urlf = "https://github.com/michalskop/ifortuna.cz/raw/master/data/MCZ26351.csv"
dff = pd.read_csv(urlf, encoding="utf-8")
rankf.append(dff.drop_duplicates(subset=['event_name'], keep='last'))
urlf = "https://github.com/michalskop/ifortuna.cz/raw/master/data/MCZ26352.csv"
dff = pd.read_csv(urlf, encoding="utf-8")
rankf.append(dff.drop_duplicates(subset=['event_name'], keep='last'))
urlf = "https://github.com/michalskop/ifortuna.cz/raw/master/data/MCZ26353.csv"
dff = pd.read_csv(urlf, encoding="utf-8")
rankf.append(dff.drop_duplicates(subset=['event_name'], keep='last'))

growf = {
  'odds': 76,
  'odds2': 88
}
for s in ['odds', 'odds2']:

  rf = [[dff['date'].iloc[-1], s]]
  for i, r in enumerate(rankf):
    # break
    rf.append([''])
    for c in dfr.columns[1:]:
      rf[i + 1].append(float(str(rankf[i][rankf[i]['event_name'] == mappingf[c]].iloc[0][s]).replace('\xa0','')))

  ws = sh.worksheet('pořadí_aktuální_aging_cov')
  ws.update('A' + str(growf[s]), rf)

  ws = sh.worksheet('pořadí_aktuální_aging')
  ws.update('A' + str(growf[s]), rf)

# Nike
urln = "https://raw.githubusercontent.com/michalskop/nike.sk/main/v0/data/42866077.csv"
dfn = pd.read_csv(urln, encoding="utf-8")

grows = {
  'áno': 103,
  'nie': 119
}
for s in ['áno', 'nie']:

  rankn = []
  rankn.append(dfn[(dfn['header'] == 'Celkovo - Víťaz') & (dfn['odds_name'] == s)].drop_duplicates(subset=['header', 'name'], keep='last'))
  rankn.append(dfn[(dfn['header'] == 'Umiestnenie do 2. miesta') & (dfn['odds_name'] == s)].drop_duplicates(subset=['header', 'name'], keep='last'))
  rankn.append(dfn[(dfn['header'] == 'Umiestnenie do 3. miesta') & (dfn['odds_name'] == s)].drop_duplicates(subset=['header', 'name'], keep='last'))
  rankn.append(dfn[(dfn['header'] == 'Umiestnenie do 4. miesta') & (dfn['odds_name'] == s)].drop_duplicates(subset=['header', 'name'], keep='last'))
  rankn.append(dfn[(dfn['header'] == 'Umiestnenie do 5. miesta') & (dfn['odds_name'] == s)].drop_duplicates(subset=['header', 'name'], keep='last'))
  rankn.append(dfn[(dfn['header'] == 'Umiestnenie do 6. miesta') & (dfn['odds_name'] == s)].drop_duplicates(subset=['header', 'name'], keep='last'))

  rn = [[dfn['date'].iloc[-1], s]]
  for i, r in enumerate(rankn):
    # break
    rn.append([''])
    for c in dfr.columns[1:]:
      rn[i + 1].append((rankn[i][rankn[i]['name'] == mappingn[c]].iloc[0]['odds']))

  ws = sh.worksheet('pořadí_aktuální_aging_cov')
  ws.update('A' + str(grows[s]), rn)

  ws = sh.worksheet('pořadí_aktuální_aging')
  ws.update('A' + str(grows[s]), rn)

# NUMBER of PARTIES
####################

# Tipsport
nint = [[dft['date'].iloc[-1], '{} a více', 'Méně než {}']]
for n in range(4, 10):
  item = ['']
  for s in ['{} a více', 'Méně než {}']:
    item.append(dft[(dft['hypername'] == 'Počet volebních subjektů v parlamentu') & (dft['name'].eq(s.format(n)))].drop_duplicates(subset=['hypername', 'supername', 'name'], keep='last')['odd'].iloc[0])
  nint.append(item)

ws = sh.worksheet('number_in')
ws.update('D23', nint)

ws = sh.worksheet('number_in_aging')
ws.update('D23', nint)

ws = sh.worksheet('number_in_aging_cov')
ws.update('D23', nint)

# Fortuna

# Nike
ninn = [[dfn['date'].iloc[-1], 'viac ako {}.5', 'menej ako {}.5']]
for n in range (6, 9):
  item = ['']
  for s in ['viac ako {}.5', 'menej ako {}.5']:
    item.append(dfn[(dfn['header'] == 'Počet volebných subjektov v parlamente') & (dfn['odds_name'].eq(s.format(n)))].drop_duplicates(subset=['header', 'name'], keep='last')['odds'].iloc[0])
  ninn.append(item)

ws = sh.worksheet('number_in')
ws.update('L26', ninn)

ws = sh.worksheet('number_in_aging')
ws.update('L26', ninn)

ws = sh.worksheet('number_in_aging_cov')
ws.update('L26', ninn)

# DUELS
####################
# Tipsport
urlt2 = "https://github.com/michalskop/tipsport.cz/raw/main/v3/data/5209621.csv"
dft2 = pd.read_csv(urlt2, encoding="utf-8")

dfft = dft2[dft2['hypername'] == 'Kdo získá více hlasů'].drop_duplicates(subset=['hypername', 'supername', 'name'], keep='last')

duelt = []
for c1 in dfr.columns[1:]:
  item = []
  for c2 in dfr.columns[1:]:
    exist = False
    filtered = dfft[(dfft['supername'] == (mappingt[c1] + ' x ' + mappingt[c2]))]
    if len(filtered) > 0:
      filtered2 = filtered[(filtered['name'] == mappingt[c1])]
      item.append(filtered2.iloc[0]['odd'])
      exist = True
    filtered = dfft[(dfft['supername'] == (mappingt[c2] + ' x ' + mappingt[c1]))]
    if len(filtered) > 0:
      filtered2 = filtered[(filtered['name'] == mappingt[c1])]
      item.append(filtered2.iloc[0]['odd'])
      exist = True
    if not exist:
      item.append('')
  duelt.append(item)
duelt.append([dfft['date'].iloc[-1]])

ws = sh.worksheet('duely_aging')
ws.update('B38', duelt)

ws = sh.worksheet('duely_aging_cov')
ws.update('B38', duelt)

# Fortuna
urlf2 = "https://github.com/michalskop/ifortuna.cz/raw/master/data/MCZ7830.csv"
dff2 = pd.read_csv(urlf2, encoding="utf-8")

dfff = dff2.drop_duplicates(subset=['event_name'], keep='last')

duelf = []
for c1 in dfr.columns[1:]:
  item = []
  for c2 in dfr.columns[1:]:
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

ws = sh.worksheet('duely_aging')
ws.update('B54', duelf)

ws = sh.worksheet('duely_aging_cov')
ws.update('B54', duelf)


# Nike
dfnf = dfn[(dfn['header'] == 'Získa viac percent')].drop_duplicates(subset=['header', 'name', 'odds_name'], keep='last')
dueln = []
for c1 in dfr.columns[1:]:
  item = []
  for c2 in dfr.columns[1:]:
    filtered = dfnf[(dfnf['name'] == (mappingn[c1] + ' - ' + mappingn[c2])) & (dfnf['odds_name'] == mappingn[c1])]
    if len(filtered) > 0:
      item.append(filtered.iloc[0]['odds'])
    else:
      filtered = dfnf[(dfnf['name'] == (mappingn[c2] + ' - ' + mappingn[c1])) & (dfnf['odds_name'] == mappingn[c1])]
      if len(filtered) > 0:
        item.append(filtered.iloc[0]['odds'])
      else:
        item.append('')
  dueln.append(item)
dueln.append([dfn['date'].iloc[-1]])

ws = sh.worksheet('duely_aging')
ws.update('B70', dueln)

ws = sh.worksheet('duely_aging_cov')
ws.update('B70', dueln)


# MORE THAN x%
####################
ws = sh.worksheet("pravděpodobnosti_aktuální_aging_cov")
dfmore = ws.col_values(1)[1:]
dfmore_end = dfmore.index('')
dfmore = dfmore[:dfmore_end]

# Tipsport
dftf = dft[dft['hypername'] == 'Počet hlasů v procentech'].drop_duplicates(subset=['hypername', 'supername', 'name'], keep='last')
last_date = dftf['date'].max()
dftf = dftf[dftf['date'] == last_date]
gcols = {
  '{}% a více': 'Q',
  'Méně než {}%': 'AD'
}
lastcol = 'AO'
# clear
ws = sh.worksheet('pravděpodobnosti_aktuální_aging_cov')
start_row = 148
range_to_clear = f'{gcols["{}% a více"]}{start_row}:{lastcol}{start_row + len(dfmore)}'
ws.batch_clear([range_to_clear])
ws = sh.worksheet('pravděpodobnosti_aktuální_aging')
ws.batch_clear([range_to_clear])

for s in ['{}% a více', 'Méně než {}%']:
  moret = []
  for n in dfmore:
    item = []
    for c in dfr.columns[1:]:
      filtered = dftf[(dftf['name'].eq(s.format(round(float(n) + 0.01, 2)))) & (dftf['supername'].eq(mappingt[c]))]
      if len(filtered) > 0:
        item.append(filtered.iloc[0]['odd'])
      else:
        item.append('')
    moret.append(item)

  ws = sh.worksheet('pravděpodobnosti_aktuální_aging_cov')
  ws.update(gcols[s] + '148', moret)

  ws = sh.worksheet('pravděpodobnosti_aktuální_aging')
  ws.update(gcols[s] + '148', moret)

# Fortuna
urlf2 = "https://raw.githubusercontent.com/michalskop/ifortuna.cz/master/data/MSK10070.v2-1.csv"
dff2 = pd.read_csv(urlf2, encoding="utf-8")

dfff = dff2.drop_duplicates(subset=['event_name', 'event_link'], keep='last')
last_date = dfff['date'].max()
dfff = dfff[dfff['date'] == last_date]
gcols = {
  'header1': 'AD',
  'header2': 'Q'
}
gstr = {
  'header1': '- {}',
  'header2': '+ {}'
}
# clear
ws = sh.worksheet('pravděpodobnosti_aktuální_aging_cov')
start_row = 221
range_to_clear = f'{gcols["header2"]}{start_row}:{lastcol}{start_row + len(dfmore)}'
ws.batch_clear([range_to_clear])
ws = sh.worksheet('pravděpodobnosti_aktuální_aging')
ws.batch_clear([range_to_clear])

i = 1
for s in ['header1', 'header2']:
  moref = []
  for n in dfmore:
    item = []
    for c in dfr.columns[1:]:
      filtered = dfff[(dfff['event_name'] == mappingf[c]) & (dfff[s].eq(gstr[s].format(float(n) + 0.01)))]
      if len(filtered) > 0:
        item.append(filtered.iloc[0]['odd' + str(i)])
      else:
        item.append('')
    moref.append(item)
  i += 1
  ws = sh.worksheet('pravděpodobnosti_aktuální_aging_cov')
  ws.update(gcols[s] + '221', moref)

  ws = sh.worksheet('pravděpodobnosti_aktuální_aging')
  ws.update(gcols[s] + '221', moref)

# Nike
gcols = {
  'viac ako': 'Q',
  'menej ako': 'AD'
}
dfnf = dfn[(dfn['header'] == 'Počet percent')].drop_duplicates(subset=['header', 'name', 'odds_name'], keep='last')
last_date = dfnf['date'].max()
dfnf = dfnf[dfnf['date'] == last_date]
# clear
ws = sh.worksheet('pravděpodobnosti_aktuální_aging_cov')
start_row = 75
range_to_clear = f'{gcols["viac ako"]}{start_row}:{lastcol}{start_row + len(dfmore)}'
ws.batch_clear([range_to_clear])
ws = sh.worksheet('pravděpodobnosti_aktuální_aging')
ws.batch_clear([range_to_clear])

for s in ['viac ako', 'menej ako']:
  moren = []
  for n in dfmore:
    item = []
    for c in dfr.columns[1:]:
      # break
      filtered = dfnf[(dfnf['name'] == mappingn[c]) & (dfnf['odds_name'].eq(s + ' ' + '{:.2f}'.format(float(n))))]
      if len(filtered) > 0:
        item.append(filtered.iloc[0]['odds'])
      else:
        item.append('')

    moren.append(item)

  ws = sh.worksheet('pravděpodobnosti_aktuální_aging_cov')
  ws.update(gcols[s] + '75', moren)

  ws = sh.worksheet('pravděpodobnosti_aktuální_aging')
  ws.update(gcols[s] + '75', moren)

