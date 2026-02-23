"""Create a new GSheet and workflow + analysis + requirements files."""

# NOTE: it stopped to create the sheet itself, so I have to create the sheet manually first and share it with all.

import os
# import sys
import datetime
import gspread
import gspread_formatting
import json
import re
import time

# Parameters for the elections
election_code = "hu-2026"
election_flag = "🇭🇺"
election_date = "2026-04-12"
source_election_code = "pt-2026" # to copy from
wikipedia_link = "https://en.wikipedia.org/wiki/Opinion_polling_for_the_2026_Hungarian_parliamentary_election"

candidates = ['Fidesz', 'Tisza', 'MH', 'DK', 'MKKP']
candidates_colors = ['#FF6A00', '#88E8FF', '#688D1B', '#2A61A4', '#808080']
candidates_values = [39, 47, 6, 3, 5] #
candidates_needs = [5, 5, 5, 5, 5] #

# SHEET KEY
sheetkey = "1a3i0HfphGxlz-6_E04wYq30tG-gGvWQcyUc4-pNwI9U"

# Create html colors by AI:
# light green in html: #3AAD2E
# light green in RGB: (58, 173, 46)

path = "./"

# pretty name, from xyz2024 to "XYZ 2024"
election_code_pretty = election_code.upper().replace(re.findall(r'\d+', election_code)[0], re.findall(r'\d+', election_code)[0])

# connect to Google Sheets
gc = gspread.service_account(path + "secret/credentials.json")

# Create a new GSheet:
# sh = gc.create(election_flag + " Výpočet chyby, simulace - " + election_code_pretty)
sh = gc.open_by_key(sheetkey)
sh.id
# Share the Gsheet with me
# Valid values are 'reader', 'commenter', 'writer', 'fileOrganizer', 'organizer', and 'owner'
# secrets = json.load(open(path + "secret/secrets.json"))
# sh.share(secrets['email'], perm_type='user', role='writer')


def hex_to_rgb(hexa):
  """Convert hex to rgb."""
  hexa = hexa.lstrip('#')
  if len(hexa) == 3:
    return tuple(int(hexa[i:i+1], 16)*17  for i in (0, 1, 2))
  else:
    return tuple(int(hexa[i:i+2], 16)  for i in (0, 2, 4))
  
def rgb_to_hex(rgb):
  """Convert rgb to hex."""
  return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def is_dark(hexa):
  """Check if the color is dark."""
  rgb = hex_to_rgb(hexa)
  if (rgb[0] + rgb[1] + rgb[2]) / 3 < 128:
    return True
  else:
    return False


# Sheet 1: info
worksheet = sh.get_worksheet(0)
# rename the sheet
worksheet.update_title('info')
time.sleep(1)
# set the values
worksheet.update('A11',[['Viz']])
worksheet.update('A12', [['https://docs.google.com/spreadsheets/d/1QCOLhcvmC04hiaFikqXGVFTtJ_dttsYxPfS-HQoK6FQ/edit#gid=850346774']], value_input_option='USER_ENTERED')
time.sleep(1)
worksheet.update('A14', [['Run:']])
worksheet.update('A15', [['https://github.com/michalskop/simple-polls/actions/workflows/run-simulations-' + election_code + '.yml']], value_input_option='USER_ENTERED')
time.sleep(1)
worksheet.update('A17', [['Wiki:']])
worksheet.update('A18', [[wikipedia_link]], value_input_option='USER_ENTERED')
time.sleep(1)
# formats
worksheet.format('A11:A12', {"textFormat": {"fontSize": 7}})
worksheet.format('A14:A15', {"textFormat": {"bold": True}, 'backgroundColor': {'red': 1, 'green': 1, 'blue': .3}})
time.sleep(1)
gspread_formatting.set_column_width(worksheet, 'A', 20)
time.sleep(1)
# flag
worksheet.update('B2', [[election_flag]])
worksheet.format('B2', {"textFormat": {"fontSize": 150}})
print("Sheet 1: info created.")

# Sheet 2: preference
worksheet = sh.add_worksheet(title="preference", rows=20, cols=20)
# worksheet = sh.worksheet("preference")
# freeze and set the first row values
worksheet.freeze(rows=1)
row = ['party', 'gain', 'date', 'volatilita', 'last update - computed (GMT)', 'needed']
worksheet.update('A1', [row])
time.sleep(1)
# set the values
for i in range(len(candidates)):
  row = [candidates[i], candidates_values[i], '', 1, '', candidates_needs[i]]
  worksheet.update('A' + str(i + 2), [row])
  # formats
  color = hex_to_rgb(candidates_colors[i])
  worksheet.format('A' + str(i + 2), {'backgroundColor': {'red': color[0]/255, 'green': color[1]/255, 'blue': color[2]/255}})
  if is_dark(candidates_colors[i]):
    worksheet.format('A' + str(i + 2), {"textFormat": {"foregroundColor": {"red": 1, "green": 1, "blue": 1}}})
  else:
    pass # default is black
  time.sleep(1)
# add dates
todate = datetime.datetime.today().isoformat()[:10]
worksheet.update('C2', [[todate]])
worksheet.format('E2', {"textFormat": {"bold": True}})
# add color
worksheet.update_tab_color(rgb_to_hex((58 / 255, 173 / 255, 46 / 255)))
print("Sheet 2: preference created.")
time.sleep(10)

# Sheet 3 + 4: pořadí aktuální
worksheet = sh.add_worksheet(title="pořadí_aktuální_aging", rows=20, cols=20)
worksheet2 = sh.add_worksheet(title="pořadí_aktuální_aging_cov", rows=20, cols=20)
# or get them by title
# worksheet = sh.worksheet("pořadí_aktuální_aging")
# worksheet2 = sh.worksheet("pořadí_aktuální_aging_cov")
time.sleep(1)
# freeze and set the first row values
worksheet.freeze(rows=1)
worksheet2.freeze(rows=1)
time.sleep(1)
row = ['Pořadí'] + candidates
worksheet.update('A1', [row])
worksheet2.update('A1', [row])
time.sleep(1)
# set the ranks
col = []
for i in range(len(candidates)):
  row = [i + 1]
  col.append(row)
  # formats first row
  color = hex_to_rgb(candidates_colors[i])
  worksheet.format(chr(66 + i) + '1', {'backgroundColor': {'red': color[0]/255, 'green': color[1]/255, 'blue': color[2]/255}})
  worksheet2.format(chr(66 + i) + '1', {'backgroundColor': {'red': color[0]/255, 'green': color[1]/255, 'blue': color[2]/255}})
  time.sleep(2)
  if is_dark(candidates_colors[i]):
    worksheet.format(chr(66 + i) + '1', {"textFormat": {"foregroundColor": {"red": 1, "green": 1, "blue": 1}}})
    worksheet2.format(chr(66 + i) + '1', {"textFormat": {"foregroundColor": {"red": 1, "green": 1, "blue": 1}}})
    time.sleep(2)
worksheet.update('A2', col)
worksheet2.update('A2', col)
time.sleep(1)
# set Fair price part
worksheet.update('A' + str(len(candidates) + 3), [['Fair price']])
worksheet2.update('A' + str(len(candidates) + 3), [['Fair price']])
time.sleep(2)
worksheet.update('A' + str(len(candidates) + 4), [['Pořadí']])
worksheet2.update('A' + str(len(candidates) + 4), [['Pořadí']])
time.sleep(2)
worksheet.update('A' + str(len(candidates) + 5), col)
worksheet2.update('A' + str(len(candidates) + 5), col)
time.sleep(2)
# set the Fair price equation
rng = []
for i in range(len(candidates)):
  row = []
  for j in range(len(candidates)):
    item = '=1/' + chr(66 + j) + (str(i + 2))
    row.append(item)
  rng.append(row)
worksheet.update('B' + str(len(candidates) + 5), rng, value_input_option='USER_ENTERED')
worksheet2.update('B' + str(len(candidates) + 5), rng, value_input_option='USER_ENTERED')
# done
print("Sheet 3 + 4: pořadí aktuální created.")
time.sleep(15)

# sheet 5 + 6: pravděpodobnosti aktuální
worksheet = sh.add_worksheet(title="pravděpodobnosti_aktuální_aging", rows=200, cols=(len(candidates) + 1))
worksheet2 = sh.add_worksheet(title="pravděpodobnosti_aktuální_aging_cov", rows=200, cols=(len(candidates) + 1))
# or get them by title
# worksheet = sh.worksheet("pravděpodobnosti_aktuální_aging")
# worksheet2 = sh.worksheet("pravděpodobnosti_aktuální_aging_cov")
time.sleep(2)
# freeze and set the first row values
worksheet.freeze(rows=1)
worksheet2.freeze(rows=1)
time.sleep(1)
row = ['Pr[zisk > x %]'] + candidates
worksheet.update('A1', [row])
worksheet2.update('A1', [row])
time.sleep(1)
# set the colors
for i in range(len(candidates)):
  color = hex_to_rgb(candidates_colors[i])
  worksheet.format(chr(66 + i) + '1', {'backgroundColor': {'red': color[0]/255, 'green': color[1]/255, 'blue': color[2]/255}})
  worksheet2.format(chr(66 + i) + '1', {'backgroundColor': {'red': color[0]/255, 'green': color[1]/255, 'blue': color[2]/255}})
  time.sleep(2)
  if is_dark(candidates_colors[i]):
    worksheet.format(chr(66 + i) + '1', {"textFormat": {"foregroundColor": {"red": 1, "green": 1, "blue": 1}}})
    worksheet2.format(chr(66 + i) + '1', {"textFormat": {"foregroundColor": {"red": 1, "green": 1, "blue": 1}}})
    time.sleep(2)
# done 
print("Sheet 5 + 6: pravděpodobnosti aktuální created.")
time.sleep(5)

# sheet 7 + 8: duely aging
worksheet = sh.add_worksheet(title="duely_aging", rows=(len(candidates) + 2), cols=(len(candidates) + 1))
worksheet2 = sh.add_worksheet(title="duely_aging_cov", rows=(len(candidates) + 2), cols=(len(candidates) + 1))
time.sleep(2)
# or get them by title
# worksheet = sh.worksheet("duely_aging")
# worksheet2 = sh.worksheet("duely_aging_cov")
# add the first cell
worksheet.update('A1', [['Pr[row >= column]']])
worksheet2.update('A1', [['Pr[row >= column]']])
time.sleep(1)
# done
print("Sheet 7 + 8: duely aging created.")
time.sleep(5)

# sheet 9 + 10: top 2 aging
worksheet = sh.add_worksheet(title="top_2", rows=(len(candidates) + 2), cols=(len(candidates) + 1))
worksheet2 = sh.add_worksheet(title="top_2_cov", rows=(len(candidates) + 2), cols=(len(candidates) + 1))
time.sleep(2)
# or get them by title
# worksheet = sh.worksheet("top_2")
# worksheet2 = sh.worksheet("top_2_cov")
# add the first cell
worksheet.update('A1', [['Pr[TOP 2]']])
worksheet2.update('A1', [['Pr[TOP 2]']])
time.sleep(1)
# done
print("Sheet 9 + 10: top 2 aging created.")
time.sleep(5)

# sheet 11: number_in_aging_cov
worksheet = sh.add_worksheet(title="number_in_aging_cov", rows=(len(candidates) + 2), cols=10)
time.sleep(1)
# add the first cell
worksheet.update('A1', [['Number in', 'Pr']])
# done
print("Sheet 11: number_in_aging_cov created.")
time.sleep(5)

# sheet 12: median correlations
worksheet = sh.add_worksheet(title="median correlations", rows=(len(candidates) + 1), cols=(len(candidates) + 1))
# or get them by title
# worksheet = sh.worksheet("median correlations")
time.sleep(1)
# add the first cell
worksheet.update('A1', [['Median']])
# fill names to row and column
worksheet.update('B1', [candidates])
col = [[x] for x in candidates]
worksheet.update('A2', col)
time.sleep(2)
# fill colors
for i in range(len(candidates)):
  color = hex_to_rgb(candidates_colors[i])
  worksheet.format('A' + str(i + 2), {'backgroundColor': {'red': color[0]/255, 'green': color[1]/255, 'blue': color[2]/255}})
  worksheet.format(chr(66 + i) + '1', {'backgroundColor': {'red': color[0]/255, 'green': color[1]/255, 'blue': color[2]/255}})
  time.sleep(2)
  if is_dark(candidates_colors[i]):
    worksheet.format('A' + str(i + 2), {"textFormat": {"foregroundColor": {"red": 1, "green": 1, "blue": 1}}})
    worksheet.format(chr(66 + i) + '1', {"textFormat": {"foregroundColor": {"red": 1, "green": 1, "blue": 1}}})
    time.sleep(2)
# fill matrix with 0 and diagonal with 1
rng = []
for i in range(len(candidates)):
  row = []
  for j in range(len(candidates)):
    if i == j:
      row.append(1)
    else:
      row.append(0)
  rng.append(row)
worksheet.update('B2', rng)
# done
print("Sheet 12: median correlations created.")

# sheet 13: history
worksheet = sh.add_worksheet(title="history", rows=20, cols=(2 * len(candidates) + 3))
time.sleep(1)
# add the first row and freeze
row = ['date_running', 'date'] + candidates + [''] + ['volatilita_' + x for x in candidates]
worksheet.update('A1', [row])
worksheet.freeze(rows=1)
time.sleep(1)
# done
print("Sheet 13: history created.")

# share with all
sh.share(None, perm_type='anyone', role='writer')

# transfer ownership to me
try:
  permissions = sh.list_permissions()
  sh.transfer_ownership(permissions[1].get('id'))
except:
  print("Transfer ownership failed.")

# CREATE THE WORKFLOW FILE
with open (path + '.github/workflows/' + 'run-simulations-' + source_election_code + '.yml') as f:
  content = f.read()
# replace the election code
content = content.replace(source_election_code, election_code)
# replace the pretty name
election_code_pretty_noflag_space = election_code.upper().replace(re.findall(r'\d+', election_code)[0], re.findall(r'\d+', election_code)[0]).replace('-', ' ')
source_election_code_pretty_noflag_space = source_election_code.upper().replace(re.findall(r'\d+', source_election_code)[0], re.findall(r'\d+', source_election_code)[0]).replace('-', ' ')
content = content.replace(source_election_code_pretty_noflag_space, election_code_pretty_noflag_space)

# save
with open (path + '.github/workflows/' + 'run-simulations-' + election_code + '.yml', 'w') as f:
  f.write(content)
print("Workflow file created.")

# CREATE REQUIREMENTS FILE
with open (path + source_election_code + '/requirements_simulations.txt') as f:
  content = f.read()
# create the directory
if not os.path.exists(path + election_code):
  os.makedirs(path + election_code)
# save
with open (path + election_code + '/requirements_simulations.txt', 'w') as f:
  f.write(content)
print("Requirements file created.")

# CREATE SIMULATIONS FILE
with open (path + source_election_code + '/simulations_' + source_election_code + '.py') as f:
  content = f.read()
# replace the election code
content = content.replace(source_election_code, election_code)
election_code_pretty_noflag = election_code.upper().replace(re.findall(r'\d+', election_code)[0], re.findall(r'\d+', election_code)[0])
source_election_code_pretty_noflag = source_election_code.upper().replace(re.findall(r'\d+', source_election_code)[0], re.findall(r'\d+', source_election_code)[0])
content = content.replace(source_election_code_pretty_noflag, election_code_pretty_noflag) 
# replace election date
content = content.replace(re.findall('election_date = \'[0-9]{4}-[0-9]{2}-[0-9]{2}\'', content)[0], 'election_date = \'' + election_date + '\'')
# replace sheetkey
content = content.replace(re.findall('sheetkey = "[a-zA-Z0-9-_]+"', content)[0], 'sheetkey = "' + sh.id + '"')
# save
with open (path + election_code + '/simulations_' + election_code + '.py', 'w') as f:
  f.write(content)
print("Simulations file created.")
