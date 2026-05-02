"""Create a new GSheet and workflow + calculator + requirements files for second round elections."""

import os
import datetime
import gspread
import gspread_formatting
import json
import re
import time

# Parameters for the elections - SECOND ROUND
election_code = "co-2026-2"
election_flag = "🇨🇴"
election_date = "2026-06-21"
source_election_code = "pt-2026-2"  # to copy from (for calculator and workflow)
wikipedia_link = "https://es.wikipedia.org/wiki/Elecciones_presidenciales_de_Colombia_de_2026"

# Races/matches for the second round
# Each race is a head-to-head match between 2 candidates
# NOTE: The 'p' (probability) values for each pair should be manually copied from:
# https://docs.google.com/spreadsheets/d/1hhRprFQjeQAFp435YUFBjqKCzxFp1q85J7TWAaWZ2JE/edit?gid=1448265129#gid=1448265129
races = [
    {
        'name': 'Cepeda vs. De La Espriella',
        'candidate1': 'Cepeda',
        'candidate2': 'De La Espriella',
        'gain1': 40.6,  # percentage for candidate 1
        'gain2': 40.5,  # percentage for candidate 2
        'color1': '#A52E94',
        'color2': '#000066'
    },
    {
        'name': 'Cepeda vs. Valencia',
        'candidate1': 'Cepeda',
        'candidate2': 'Valencia',
        'gain1': 45,  # percentage for candidate 1
        'gain2': 45,  # percentage for candidate 2
        'color1': '#A52E94',
        'color2': '#63B9E9'
    },
    {
        'name': 'De La Espriella vs. Valencia',
        'candidate1': 'De La Espriella',
        'candidate2': 'Valencia',
        'gain1': 45,  # percentage for candidate 1
        'gain2': 45,  # percentage for candidate 2
        'color1': '#000066',
        'color2': '#63B9E9'
    }
    # Add more races as needed
]

# Additional limits for probability calculations (optional)
# These are specific gain values where you want to calculate probabilities
additional_limits = []  # e.g., [26.33, 22.79, 17.11]

# Common parameters for all races (can be overridden per race)
# NOTE: We assume that gain1 + gain2 <= 100 (the rest is "others" or undecided)
default_params = {
    'election_date': election_date,  # ISO date of the election
    'current_date': datetime.datetime.today().isoformat()[:10],  # ISO date for which the model is calculated (often "today")
    'sample_n': 1000,  # Sample size used in statistical error calculation
    're_coef': 1,  # Random error coefficient
    'aging_coef': 1.15,  # Aging coefficient: aging_coeff = pow(diff, aging_coef) / diff
    'sample': 10000,  # Number of simulations to run
    'interval_min': 0,  # Lowest gain percentage to calculate probability for
    'interval_max': 100,  # Highest gain percentage to calculate probability for
    'step': 0.5,  # Step size for probability calculations
    'volatilita': 1  # Added volatility parameter
}

# SHEET KEY - create the sheet manually first and paste the key here
sheetkey = "1_z4CFUxqRzomGVHXmS-2MfDzN-xQX-NGmLkhT44-7zg"

path = "./"

# pretty name, from xyz2024-2 to "XYZ 2024 2"
election_code_pretty = election_code.upper().replace(re.findall(r'\d+', election_code)[0], re.findall(r'\d+', election_code)[0])

# connect to Google Sheets
gc = gspread.service_account(path + "secret/credentials.json")

sh = gc.open_by_key(sheetkey)
sh.id


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


def get_or_create_worksheet(spreadsheet, title, rows=100, cols=20):
  """Get existing worksheet or create new one if it doesn't exist."""
  try:
    worksheet = spreadsheet.worksheet(title)
    print(f"Sheet '{title}' already exists, using existing sheet.")
    return worksheet
  except gspread.exceptions.WorksheetNotFound:
    worksheet = spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)
    print(f"Sheet '{title}' created.")
    return worksheet


# Sheet 1: info
worksheet = sh.get_worksheet(0)
worksheet.update_title('info')
time.sleep(1)
worksheet.update(range_name='A11', values=[['Viz']])
worksheet.update(range_name='A12', values=[['https://docs.google.com/spreadsheets/d/1QCOLhcvmC04hiaFikqXGVFTtJ_dttsYxPfS-HQoK6FQ/edit#gid=850346774']], value_input_option='USER_ENTERED')
time.sleep(1)
worksheet.update(range_name='A14', values=[['Run:']])
worksheet.update(range_name='A15', values=[['https://github.com/michalskop/simple-polls/actions/workflows/multicalculator-' + election_code + '.yml']], value_input_option='USER_ENTERED')
time.sleep(1)
worksheet.update(range_name='A17', values=[['Wiki:']])
worksheet.update(range_name='A18', values=[[wikipedia_link]], value_input_option='USER_ENTERED')
time.sleep(1)
worksheet.format('A11:A12', {"textFormat": {"fontSize": 7}})
worksheet.format('A14:A15', {"textFormat": {"bold": True}, 'backgroundColor': {'red': 1, 'green': 1, 'blue': .3}})
time.sleep(1)
gspread_formatting.set_column_width(worksheet, 'A', 20)
time.sleep(1)
worksheet.update(range_name='B2', values=[[election_flag]])
worksheet.format('B2', {"textFormat": {"fontSize": 150}})
print("Sheet 1: info created.")

# Sheet 2: parametry (main parameters for each race)
worksheet = get_or_create_worksheet(sh, "parametry", rows=(len(races) + 20), cols=20)
time.sleep(1)
worksheet.freeze(rows=1)

# Headers
headers = ['p', 'name', 'gain1', 'gain2', 'election date', 'current date', 'volatilita', 
           'aging coef', 'sample n', 're coef', 'sample', 'interval min', 'interval max', 'step', 
           'x', 'last successful calculation (GMT)', 'y', 'notes']
worksheet.update(range_name='A1', values=[headers])
time.sleep(1)

# Row 2: First race data in A-D, default parameters in E-N
# Write default parameters to E2:N2 (election date through step)
worksheet.update(range_name='E2:N2', values=[
    [default_params['election_date'], default_params['current_date'], default_params['volatilita'],
     default_params['aging_coef'], default_params['sample_n'], default_params['re_coef'],
     default_params['sample'], default_params['interval_min'], default_params['interval_max'],
     default_params['step']]
])
time.sleep(1)

# Fill in race data starting from row 2
start_row = 2
for i, race in enumerate(races):
    # Write only the race-specific data (p, name, gain1, gain2) to columns A-D
    race_data = [
        [race.get('p', ''), race['name'], race['gain1'], race['gain2']]
    ]
    worksheet.update(range_name='A' + str(start_row + i) + ':D' + str(start_row + i), values=race_data)
    time.sleep(1)

# Add notes documentation in column R starting at row 1
notes_start_row = 1
notes_lines = [
    ['notes'],
    ['počítáme, že součet je <= 100'],
    ['current date ~ iso datum, ke kterému se model počítá (často "dnešní datum")'],
    ['volatilita ~ přidaná volatilita'],
    ['aging_coeff = pow(diff, aging coef) / diff'],
    ['sample n ~ used in statistical error'],
    ['re coef ~ random error coefficient'],
    ['sample ~ number of simulations'],
    ['interval min ~ lowest gain to calc probability'],
    ['interval max ~ highest gain to calc probability'],
    ['step ~ to calc probability'],
    ['p ~ pravděpodobnost této dvojice, zkopírované ručně']
]
worksheet.update(range_name='R' + str(notes_start_row), values=notes_lines)
time.sleep(1)

# Sheet 3: parametry 2 (additional limits)
worksheet = get_or_create_worksheet(sh, "parametry 2", rows=20, cols=5)
time.sleep(1)
worksheet.freeze(rows=1)
worksheet.update(range_name='A1', values=[['additional limits']])
time.sleep(1)
if additional_limits:
    for i, limit in enumerate(additional_limits):
        worksheet.update(range_name='A' + str(i + 2), values=[[limit]])
        time.sleep(1)

# Sheet 4: poradi (rankings/winning probabilities)
worksheet = get_or_create_worksheet(sh, "poradi", rows=(len(races) + 10), cols=5)
time.sleep(1)
worksheet.freeze(rows=1)
worksheet.update(range_name='A1', values=[['Pr[duel winned]', 'p1', 'p2']])
time.sleep(1)

# Sheet 5 & 6: pravdepodobnosti1 and pravdepodobnosti2 (probabilities for each candidate)
for j in range(1, 3):
    worksheet = get_or_create_worksheet(sh, "pravdepodobnosti" + str(j), rows=300, cols=(len(races) + 5))
    time.sleep(1)
    worksheet.freeze(rows=1)
    worksheet.update(range_name='A1', values=[['Pr[duel zisk > x %]']])
    time.sleep(1)

# Sheet 7: difference (difference between candidates)
worksheet = get_or_create_worksheet(sh, "difference", rows=100, cols=(len(races) + 5))
time.sleep(1)
worksheet.freeze(rows=1)
worksheet.update(range_name='A1', values=[['Pr[difference(X1-X2) > x]']])
time.sleep(1)

# Sheet 8: history
worksheet = get_or_create_worksheet(sh, "history", rows=100, cols=10)
time.sleep(1)
worksheet.freeze(rows=1)
headers = ['date_running', 'date', 'match', 'gain', 'value', 'volatilita']
worksheet.update(range_name='A1', values=[headers])
time.sleep(1)

# share with all
sh.share(None, perm_type='anyone', role='writer')

# transfer ownership to me
try:
  permissions = sh.list_permissions()
  sh.transfer_ownership(permissions[1].get('id'))
except:
  print("Transfer ownership failed.")

# CREATE THE WORKFLOW FILE
# Try workflows/ directory first, then workflows-2026-02/ as fallback
workflow_source_path = path + '.github/workflows/' + 'multicalculator-' + source_election_code + '.yml'
if not os.path.exists(workflow_source_path):
    workflow_source_path = path + '.github/workflows-2026-02/' + 'multicalculator-' + source_election_code + '.yml'

with open(workflow_source_path) as f:
  content = f.read()

# replace the election code
content = content.replace(source_election_code, election_code)

# replace the pretty name
election_code_pretty_noflag_space = election_code.upper().replace(re.findall(r'\d+', election_code)[0], re.findall(r'\d+', election_code)[0]).replace('-', ' ')
source_election_code_pretty_noflag_space = source_election_code.upper().replace(re.findall(r'\d+', source_election_code)[0], re.findall(r'\d+', source_election_code)[0]).replace('-', ' ')
content = content.replace(source_election_code_pretty_noflag_space, election_code_pretty_noflag_space)

# save to workflows/ directory
workflow_dir = path + '.github/workflows/'
if not os.path.exists(workflow_dir):
    os.makedirs(workflow_dir)
    
with open(workflow_dir + 'multicalculator-' + election_code + '.yml', 'w') as f:
  f.write(content)
print("Workflow file created.")

# CREATE REQUIREMENTS FILE
requirements_source_path = path + source_election_code + '/requirements-calculator.txt'
if not os.path.exists(requirements_source_path):
    print(f"Warning: Source requirements file not found at {requirements_source_path}")
    # Create a default requirements file
    requirements_content = """gspread
gspread-formatting
numpy
pandas
scipy
"""
else:
    with open(requirements_source_path) as f:
        requirements_content = f.read()

# create the directory
if not os.path.exists(path + election_code):
  os.makedirs(path + election_code)

# save
with open(path + election_code + '/requirements-calculator.txt', 'w') as f:
  f.write(requirements_content)
print("Requirements file created.")

# CREATE CALCULATOR FILE
calculator_source_path = path + source_election_code + '/calculator.py'
if not os.path.exists(calculator_source_path):
    print(f"Error: Source calculator file not found at {calculator_source_path}")
else:
    with open(calculator_source_path) as f:
        content = f.read()
    
    # replace the sheetkey
    content = re.sub(r'sheetkey = "[a-zA-Z0-9-_]+"', 'sheetkey = "' + sh.id + '"', content)
    
    # replace the path
    content = re.sub(r'path = "' + source_election_code + '/"', 'path = "' + election_code + '/"', content)
    
    # save
    with open(path + election_code + '/calculator.py', 'w') as f:
        f.write(content)
    print("Calculator file created.")

print("\n" + "="*50)
print("SETUP COMPLETE!")
print("="*50)
print(f"Election: {election_code}")
print(f"Sheet ID: {sh.id}")
print(f"Sheet URL: https://docs.google.com/spreadsheets/d/{sh.id}")
print(f"Directory created: {election_code}/")
print(f"Workflow file: .github/workflows-2026-02/multicalculator-{election_code}.yml")
print("="*50)
