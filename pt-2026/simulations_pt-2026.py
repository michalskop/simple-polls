"""Simulations for PT-2025."""

import datetime
import gspread
import math
import numpy as np
import pandas as pd
import scipy.stats
import warnings
# from matplotlib import pyplot as plt

election_date = '2026-01-18'
election_day = datetime.date.fromisoformat(election_date)
today = datetime.date.today()   # it changes later !!!
sample_n = 1000 # used in statistical error
re_coef = 0.6 # random error coefficient
sample = 2000 # number of simulation
interval_max = 60 # highest gain to calc probability
# source sheet
sheetkey = "1hhRprFQjeQAFp435YUFBjqKCzxFp1q85J7TWAaWZ2JE"
path = "pt-2026/"

# additional_points = [0.55, 1.11]
# additional_points = [2.9, 4.14, 6.34, 7.3, 10.94, 11.54, 12.74, 19.54, 20.54, 20.74] # + 0.01
additional_points = []

# load data from GSheet
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)

ws = sh.worksheet('preference')
dfpreference = pd.DataFrame(ws.get_all_records())
dfpreference['p'] = dfpreference['gain'] / 100
# today
today = datetime.date.fromisoformat(dfpreference['date'][0])

# aging curve 
def aging_coeff(day1, day2):
  diff = abs((day2 - day1).days)
  if diff <= 0:
    return 1
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
  p = normal_error(dfpreference, sample_n, dfpreference['volatilita'], 0.9)
  p = uniform_error(p, sample_n, dfpreference['volatilita'], 1.5 * 0.9 * 0.9)
  p['estimate'] = p['normal_error'] + p['uniform_error'] + p['p']
  p['estimate_aging'] = aging * (p['normal_error'] + p['uniform_error']) + p['p']
  simx = dict(zip(dfpreference['party'].to_list(), p['estimate']))
  simxa = dict(zip(dfpreference['party'].to_list(), p['estimate_aging']))
  # simulations = simulations.append(simx, ignore_index=True)
  simulations = pd.concat([simulations, pd.DataFrame([simx])], ignore_index=True)
  # simulations_aging = simulations_aging.append(simxa, ignore_index=True)
  simulations_aging = pd.concat([simulations_aging, pd.DataFrame([simxa])], ignore_index=True)

# simulations with correlations
# note: correlation is used only for the normal distribution part
wsc = sh.worksheet('median correlations')
correlations = pd.DataFrame(wsc.get_all_records())
# reorder to match p
t = p.loc[:, ['party']].merge(correlations, left_on='party', right_on='Median')
del t['party']
tt = t.loc[:, ['Median']]
for c in tt['Median']:
  tt[c] = t.loc[:, c]
del tt['Median']
# simulations
corr = tt.to_numpy()
cov = p['sdx'].to_numpy() * corr * p['sdx'].to_numpy().T
try:
  simulations_cov = np.random.multivariate_normal(mean=p['p'], cov=cov, size=sample)
except RuntimeWarning as warning:
  print('Covariance matrix is not positive definite.')
  # Ensure the covariance matrix is positive-semidefinite
  eigenvalues, eigenvectors = np.linalg.eigh(cov)
  eigenvalues = np.maximum(eigenvalues, 0)  # Set negative eigenvalues to zero
  cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T  # Reconstruct the covariance matrix
  simulations_cov = np.random.multivariate_normal(mean=p['p'], cov=cov, size=sample)

p['sdxage'] = p['sdx'] * aging
covage = p['sdxage'].to_numpy() * corr * p['sdxage'].to_numpy().T
simulation_aging_cov = np.random.multivariate_normal(mean=p['p'], cov=covage, size=sample)
simulations_cov = pd.DataFrame(simulations_cov, columns=dfpreference['party'].to_list())
simulations_aging_cov = pd.DataFrame(simulation_aging_cov, columns=dfpreference['party'].to_list())
# add uniform error
for c in simulations_cov.columns:
  sx = p[p['party'] == c]['sdx'].values[0]
  simulations_cov[c] = simulations_cov[c] + np.random.uniform(low=(-1 * sx * math.sqrt(3)), high=(sx * math.sqrt(3)), size=sample)
  simulations_aging_cov[c] = simulations_aging_cov[c] + np.random.uniform(low=(-1 * sx * aging * math.sqrt(3)), high=(sx * aging * math.sqrt(3)), size=sample)

# rank matrix (somehow did not work directly)
ranks = simulations.loc[0:sample,:].rank(axis=1, ascending=False)
ranks_statistics = pd.DataFrame(index=ranks.columns)
ranks_aging = simulations_aging.loc[0:sample,:].rank(axis=1, ascending=False)
ranks_statistics_aging = pd.DataFrame(index=ranks_aging.columns)
for i in range(1, len(ranks.columns) + 1):
  ranks_statistics[str(i)] = pd.DataFrame((ranks <= i).sum() / sample).rename(columns={0: str(i)})
  ranks_statistics_aging[str(i)] = pd.DataFrame((ranks_aging <= i).sum() / sample).rename(columns={0: str(i)})

# top 2
top2 = ranks_aging.where(ranks_aging <= 2).fillna(False).where(ranks_aging > 2).fillna(True)
top2_statistics = pd.DataFrame(index=ranks_aging.columns, columns=ranks_aging.columns)
for i in range(0, len(ranks_aging.columns)):
  for j in range(0, len(ranks_aging.columns)):
    if i != j:
      top2_statistics.iloc[i, j] = (top2.iloc[:, i] & top2.iloc[:, j]).sum() / sample
    else:
      top2_statistics.iloc[i, j] = ''

# rank matrix (somehow did not work directly) - covariances
ranks_cov = simulations_cov.loc[0:sample,:].rank(axis=1, ascending=False)
ranks_statistics_cov = pd.DataFrame(index=ranks_cov.columns)
ranks_aging_cov = simulations_aging_cov.loc[0:sample,:].rank(axis=1, ascending=False)
ranks_statistics_aging_cov = pd.DataFrame(index=ranks_aging_cov.columns)
for i in range(1, len(ranks_cov.columns)):
  ranks_statistics_cov[str(i)] = pd.DataFrame((ranks_cov <= i).sum() / sample).rename(columns={0: str(i)})
  ranks_statistics_aging_cov[str(i)] = pd.DataFrame((ranks_aging_cov <= i).sum() / sample).rename(columns={0: str(i)})

# rank matrix (somehow did not work directly) - covariances
# to number of seats, if the same number, then the same rank, the worse one
ranks_cov_seats = ((simulations_cov * 150).round().loc[0:sample,:].rank(axis=1, ascending=False) + 0.45).round()
ranks_statistics_cov_seats = pd.DataFrame(index=ranks_cov_seats.columns)
ranks_aging_cov_seats = ((simulations_aging_cov * 150).round().loc[0:sample,:].rank(axis=1, ascending=False) + 0.45).round()
ranks_statistics_aging_cov_seats = pd.DataFrame(index=ranks_aging_cov_seats.columns)
for i in range(1, len(ranks_cov_seats.columns)):
  ranks_statistics_cov_seats[str(i)] = pd.DataFrame((ranks_cov_seats <= i).sum() / sample).rename(columns={0: str(i)})
  ranks_statistics_aging_cov_seats[str(i)] = pd.DataFrame((ranks_aging_cov_seats <= i).sum() / sample).rename(columns={0: str(i)})

# top 2
top2_cov = ranks_aging_cov.where(ranks_aging_cov <= 2).fillna(False).where(ranks_aging_cov > 2).fillna(True)
top2_statistics_cov = pd.DataFrame(index=ranks_aging_cov.columns, columns=ranks_aging_cov.columns)
for i in range(0, len(ranks_aging_cov.columns)):
  for j in range(0, len(ranks_aging_cov.columns)):
    if i != j:
      top2_statistics_cov.iloc[i, j] = (top2_cov.iloc[:, i] & top2_cov.iloc[:, j]).sum() / sample
    else:
      top2_statistics_cov.iloc[i, j] = ''

# less than
arr = np.concatenate((np.arange(0, interval_max + 0.5, 0.5), np.array(additional_points)))

interval_statistics = pd.DataFrame(columns=dfpreference['party'].to_list())
interval_statistics_aging = pd.DataFrame(columns=dfpreference['party'].to_list())
interval = pd.DataFrame(columns=['Pr'])
for i in arr:
# for i in np.concatenate((np.arange(0, interval_max + 0.5, 0.5), np.array([]))):    
  # interval = interval.append({'Pr': i}, ignore_index=True)
  interval = pd.concat([interval, pd.DataFrame({'Pr': i}, index=[0])], ignore_index=True)
  # interval_statistics = interval_statistics.append((simulations > (i / 100)).sum() / sample, ignore_index=True)
  interval_statistics = pd.concat(
    [interval_statistics, 
    pd.DataFrame([(simulations > (i / 100)).sum() / sample], columns=dfpreference['party'].to_list())
    ], ignore_index=True
  )
  # interval_statistics_aging = interval_statistics_aging.append((simulations_aging > (i / 100)).sum() / sample, ignore_index=True)
  interval_statistics_aging = pd.concat([interval_statistics_aging, pd.DataFrame([(simulations_aging > (i / 100)).sum() / sample], columns=dfpreference['party'].to_list())], ignore_index=True)

# less than covariance
interval_statistics_cov = pd.DataFrame(columns=dfpreference['party'].to_list())
interval_statistics_aging_cov = pd.DataFrame(columns=dfpreference['party'].to_list())
interval_cov = pd.DataFrame(columns=['Pr'])
for i in arr:
# for i in np.concatenate((np.arange(0, interval_max + 0.5, 0.5), np.array([]))):    
  # interval_cov = interval_cov.append({'Pr': i}, ignore_index=True)
  interval_cov = pd.concat([interval_cov, pd.DataFrame({'Pr': i}, index=[0])], ignore_index=True)
  # interval_statistics_cov = interval_statistics_cov.append((simulations_cov > (i / 100)).sum() / sample, ignore_index=True)
  interval_statistics_cov = pd.concat([interval_statistics_cov, pd.DataFrame([(simulations_cov > (i / 100)).sum() / sample], columns=dfpreference['party'].to_list())], ignore_index=True)
  # interval_statistics_aging_cov = interval_statistics_aging_cov.append((simulations_aging_cov > (i / 100)).sum() / sample, ignore_index=True)
  interval_statistics_aging_cov = pd.concat([interval_statistics_aging_cov, pd.DataFrame([(simulations_aging_cov > (i / 100)).sum() / sample], columns=dfpreference['party'].to_list())], ignore_index=True)

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
duels_aging_cov = pd.DataFrame(columns = ranks_aging_cov.columns, index=ranks_aging_cov.columns)
for i in ranks_aging_cov.columns:
  for j in ranks_aging_cov.columns:
    p = (sum(ranks_aging_cov[i] >= ranks_aging_cov[j])) / sample
    duels_aging_cov[i][j] = p

# number of parties in parliament
needed = dfpreference.loc[:, ['party', 'needed']].set_index('party')

number_in_sim = simulations.T.ge(needed['needed'], axis=0).sum().to_frame().rename(columns={0: 'number_in'})
nic = number_in_sim.value_counts(sort=False, ascending=True)
number_in = pd.DataFrame(index=range(0, number_in_sim['number_in'].max() + 1), columns=['p'])
for i in range(0, nic.index.max()[0] + 1):
  number_in['p'][i] = nic.loc[i:].sum() / sample

# number of parties in parliament - aging
number_in_sim_aging = simulations_aging.T.ge(needed['needed'], axis=0).sum().to_frame().rename(columns={0: 'number_in'})
nic_aging = number_in_sim_aging.value_counts(sort=False, ascending=True)
number_in_aging = pd.DataFrame(index=range(0, number_in_sim_aging['number_in'].max() + 1), columns=['p'])
for i in range(0, nic_aging.index.max()[0] + 1):
  number_in_aging['p'][i] = nic_aging.loc[i:].sum() / sample

# number of parties in parliament - aging - cov
number_in_sim_aging_cov = simulations_aging_cov.T.ge(needed['needed'], axis=0).sum().to_frame().rename(columns={0: 'number_in'})
nic_aging_cov = number_in_sim_aging_cov.value_counts(sort=False, ascending=True)
number_in_aging_cov = pd.DataFrame(index=range(0, number_in_sim_aging_cov['number_in'].max() + 1), columns=['p'])
for i in range(0, nic_aging_cov.index.max()[0] + 1):
  number_in_aging_cov['p'][i] = nic_aging_cov.loc[i:].sum() / sample

# WRITE TO SHEET
# wsw = sh.worksheet('pořadí_aktuální')
# wsw.update('B1', [ranks_statistics.transpose().columns.values.tolist()] + ranks_statistics.transpose().values.tolist())

wsw = sh.worksheet('pořadí_aktuální_aging')
wsw.update([ranks_statistics_aging.transpose().columns.values.tolist()] + ranks_statistics_aging.transpose().values.tolist(), range_name='B1')

wsw = sh.worksheet('pořadí_aktuální_aging_cov')
wsw.update([ranks_statistics_aging_cov.transpose().columns.values.tolist()] + ranks_statistics_aging_cov.transpose().values.tolist(), range_name='B1')

# wsw = sh.worksheet('pořadí_aktuální_aging_cov_seats')
# wsw.update('B1', [ranks_statistics_aging_cov_seats.transpose().columns.values.tolist()] + ranks_statistics_aging_cov_seats.transpose().values.tolist())

# wsw = sh.worksheet('pravděpodobnosti_aktuální')
# wsw.update('B1', [interval_statistics.columns.values.tolist()] + interval_statistics.values.tolist())

wsw = sh.worksheet('pravděpodobnosti_aktuální_aging')
arr2 = []
for item in arr:
  arr2.append([item])
wsw.update(arr2, range_name='A2')
wsw.update([interval_statistics_aging.columns.values.tolist()] + interval_statistics_aging.values.tolist(), range_name='B1')

wsw = sh.worksheet('pravděpodobnosti_aktuální_aging_cov')
wsw.update(arr2, range_name='A2')
wsw.update([interval_statistics_aging_cov.columns.values.tolist()] + interval_statistics_aging_cov.values.tolist(), range_name='B1')

# wsw = sh.worksheet('duely')
# wsw.update('B2', [duels.columns.values.tolist()] + duels.values.tolist())

wsw = sh.worksheet('duely_aging')
arrd = []
for item in duels_aging.columns:
  arrd.append([item])
wsw.update(arrd, range_name='A3')
wsw.update([duels_aging.columns.values.tolist()] + duels_aging.values.tolist(), range_name='B2')

wsw = sh.worksheet('duely_aging_cov')
arrd = []
for item in duels_aging_cov.columns:
  arrd.append([item])
wsw.update(arrd, range_name='A3')
wsw.update([duels_aging_cov.columns.values.tolist()] + duels_aging_cov.values.tolist(), range_name='B2')

wsw = sh.worksheet('top_2')
wsw.update(arrd, range_name='A3')
wsw.update([top2_statistics.columns.values.tolist()] + top2_statistics.values.tolist(), range_name='B2')

wsw = sh.worksheet('top_2_cov')
wsw.update(arrd, range_name='A3')
wsw.update([top2_statistics_cov.columns.values.tolist()] + top2_statistics_cov.values.tolist(), range_name='B2')

# wsw = sh.worksheet('number_in')
# number_in = number_in.reset_index(drop=False)
# wsw.update('A2', number_in.values.tolist())

# wsw = sh.worksheet('number_in_aging')
# number_in_aging = number_in_aging.reset_index(drop=False)
# wsw.update('A2', number_in_aging.values.tolist())

wsw = sh.worksheet('number_in_aging_cov')
number_in_aging_cov = number_in_aging_cov.reset_index(drop=False)
wsw.update(values=number_in_aging_cov.values.tolist(), range_name='A2')

wsw = sh.worksheet('preference')
d = datetime.datetime.now().isoformat()
wsw.update(values=[[d]], range_name='E2')

# --- New functionality: Victory Margin Calculation ---

print("\nCalculating victory margin probabilities...")

try:
    # Get the worksheet for victory margins
    victory_ws = sh.worksheet('victory_margin_aging_cov')
    
    # Find columns with "lower" and "upper" headers dynamically
    header_row = victory_ws.get('1:1')[0]
    lower_col_idx = None
    upper_col_idx = None
    
    for i, header in enumerate(header_row):
        if header and 'lower' in str(header).lower():
            lower_col_idx = i
        if header and 'upper' in str(header).lower():
            upper_col_idx = i
    
    if lower_col_idx is None or upper_col_idx is None:
        print("❌ Error: Could not find 'lower' and/or 'upper' columns in victory_margin_aging_cov sheet")
        print("Skipping victory margin calculation")
    else:
        # Helper function to convert column number to letter
        def num_to_col(num):
            result = ""
            while num > 0:
                num -= 1
                result = chr(65 + (num % 26)) + result
                num //= 26
            return result
        
        lower_col_letter = num_to_col(lower_col_idx + 1)
        upper_col_letter = num_to_col(upper_col_idx + 1)
        
        # Get intervals from lower and upper columns
        lower_values = victory_ws.get(f'{lower_col_letter}2:{lower_col_letter}50')
        upper_values = victory_ws.get(f'{upper_col_letter}2:{upper_col_letter}50')
        
        # Combine into intervals
        intervals = []
        max_rows = max(len(lower_values), len(upper_values))
        for i in range(max_rows):
            try:
                lower_val = float(lower_values[i][0]) if i < len(lower_values) and len(lower_values[i]) > 0 and lower_values[i][0] else None
                upper_val = float(upper_values[i][0]) if i < len(upper_values) and len(upper_values[i]) > 0 and upper_values[i][0] else None
                if lower_val is not None and upper_val is not None:
                    intervals.append((lower_val, upper_val))
            except (ValueError, TypeError, IndexError):
                continue
        
        print(f"Found {len(intervals)} victory margin intervals")
        
        # Get candidate names from the sheet headers (columns C-G typically)
        # Get headers from C1 to G1 (or wider range if needed)
        candidate_headers = victory_ws.get('C1:G1')[0]
        
        # Create mapping from candidate name to column letter
        candidate_name_to_col = {}
        for i, header in enumerate(candidate_headers):
            if header and header.strip():
                col_letter = num_to_col(3 + i)  # C is column 3
                candidate_name_to_col[header] = col_letter
        
        print(f"Found candidate columns: {candidate_name_to_col}")
        
        # Get candidate names from simulations (these are the actual party names)
        simulation_candidates = simulations_aging_cov.columns.tolist()
        print(f"Simulation candidates: {simulation_candidates}")
        
        if not candidate_name_to_col or not intervals:
            print("❌ Error: Could not find candidate columns or intervals")
            print("Skipping victory margin calculation")
        else:
            # Initialize a DataFrame to store the counts
            # Use the candidate names from the sheet headers as columns
            sheet_candidate_names = list(candidate_name_to_col.keys())
            victory_margin_counts = pd.DataFrame(0, index=range(len(intervals)), columns=sheet_candidate_names)
            
            # Iterate through each simulation run
            for i in range(sample):
                # Get the results for the current simulation
                run = simulations_aging_cov.iloc[i].sort_values(ascending=False)
                
                # Identify winner and runner-up
                winner_name = run.index[0]
                winner_score = run.iloc[0]
                runner_up_score = run.iloc[1]
                
                # Calculate margin of victory in percentage points
                margin = (winner_score - runner_up_score) * 100
                
                # Check if the winner matches any candidate name in the sheet
                # Try exact match first, then partial match
                matched_candidate = None
                for sheet_candidate in sheet_candidate_names:
                    if winner_name == sheet_candidate or winner_name in str(sheet_candidate) or str(sheet_candidate) in winner_name:
                        matched_candidate = sheet_candidate
                        break
                
                if matched_candidate:
                    # Find which intervals the margin falls into (cumulative - can match multiple intervals)
                    for interval_idx, (lo, hi) in enumerate(intervals):
                        # Handle special case where hi >= 100 means "at least lo"
                        if hi >= 100:
                            if margin >= lo:
                                victory_margin_counts.loc[interval_idx, matched_candidate] += 1
                        else:
                            # Normal interval: margin must be >= lo and < hi
                            if lo <= margin < hi:
                                victory_margin_counts.loc[interval_idx, matched_candidate] += 1
            
            # Calculate probabilities
            victory_margin_probabilities = victory_margin_counts / sample
            
            # Write each candidate's column separately to avoid overwriting
            print(f"Writing victory margin probabilities to columns: {list(candidate_name_to_col.values())}")
            try:
                for candidate_name, col_letter in candidate_name_to_col.items():
                    if candidate_name in victory_margin_probabilities.columns:
                        col_data = [[val] for val in victory_margin_probabilities[candidate_name].values]
                        update_range = f"{col_letter}2:{col_letter}{2 + len(intervals) - 1}"
                        victory_ws.update(col_data, range_name=update_range)
                        print(f"  ✓ Wrote {candidate_name} probabilities to column {col_letter}")
                print("✓ Victory margin probabilities written successfully.")
            except Exception as e:
                print(f"❌ Error writing victory margin probabilities: {e}")

except Exception as e:
    print(f"❌ Error during victory margin calculation: {e}")
    import traceback
    traceback.print_exc()

# save to history initial preferences
historical_row = [d] + [dfpreference['date'][0]] + dfpreference['gain'].to_list() + [''] + dfpreference['volatilita'].to_list()
wsh = sh.worksheet('history')
wsh.insert_row(historical_row, 2)

# save to history
# ranks
# history = pd.read_csv(path + 'history_1_rank.csv')
# newly = pd.DataFrame(columns=history.columns)
# cols = ranks_statistics.T.columns
# for col in cols:
#   t = ranks_statistics.T[col].to_frame().reset_index().rename(columns={'index': 'rank', col: 'p'})
#   t['gain'] = dfpreference[dfpreference['party'] == col]['gain'].values[0]
#   t['name'] = col
#   t['datetime'] = d
#   t['date'] = today.isoformat()
#   # newly = newly.append(t, ignore_index=True)
#   newly = pd.concat([newly, pd.DataFrame(t, columns=history.columns)])

# pd.concat([history, newly], ignore_index=True).to_csv(path + 'history_1_rank.csv', index=False)

# # probability
# history = pd.read_csv(path + 'history_1_prob.csv')
# newly = pd.DataFrame(columns=history.columns)
# cols = interval_statistics.columns
# for col in cols:
#     t = interval_statistics[col].to_frame()
#     t.columns = ['p']
#     t['less'] = interval['Pr']
#     t['datetime'] = d
#     t['gain'] = dfpreference[dfpreference['party'] == col]['gain'].values[0]
#     t['name'] = col
#     t['date'] = today.isoformat()
#     # newly = newly.append(t, ignore_index=True)
#     newly = pd.concat([newly, pd.DataFrame(t, columns=history.columns)])

# pd.concat([history, newly], ignore_index=True).to_csv(path + 'history_1_prob.csv', index=False)

# # top2
# history = pd.read_csv(path + 'history_1_top2.csv')
# newly = pd.DataFrame(columns=history.columns)
# cols = top2_statistics.columns
# for col in cols:
#   for row in cols:
#     if row > col:
#       t = {}
#       t['p'] = top2_statistics[col][row]
#       t['name1'] = col
#       t['name2'] = row
#       t['gain1'] = dfpreference[dfpreference['party'] == col]['gain'].values[0]
#       t['gain2'] = dfpreference[dfpreference['party'] == row]['gain'].values[0]
#       t['date'] = today.isoformat()
#       t['datetime'] = d
      
#       # newly = newly.append(t, ignore_index=True)
#       newly = pd.concat([newly, pd.DataFrame(t, columns=history.columns, index=[0])])

# pd.concat([history, newly], ignore_index=True).to_csv(path + 'history_1_top2.csv', index=False)

# # duels 1
# history = pd.read_csv(path + 'history_1_duel.csv')
# newly = pd.DataFrame(columns=history.columns)
# cols = duels_aging.columns
# for col in cols:
#   for row in cols:
#     if row > col:
#       t = {}
#       t['p'] = duels_aging[row][col]
#       t['name1'] = col
#       t['name2'] = row
#       t['gain1'] = dfpreference[dfpreference['party'] == col]['gain'].values[0]
#       t['gain2'] = dfpreference[dfpreference['party'] == row]['gain'].values[0]
#       t['date'] = today.isoformat()
#       t['datetime'] = d
      
#       # newly = newly.append(t, ignore_index=True)
#       newly = pd.concat([newly, pd.DataFrame(t, columns=history.columns, index=[0])])

# pd.concat([history, newly], ignore_index=True).to_csv(path + 'history_1_duel.csv', index=False)
