"""Simulations for PT-2025."""

import datetime
import gspread
import math
import numpy as np
import pandas as pd
import scipy.stats
import warnings
from allocate_seats_hungary import HungarianSeatAllocator
from allocate_seats_hungary_v2 import HungarianSeatAllocatorV2
# from matplotlib import pyplot as plt

election_date = '2026-04-12'
election_day = datetime.date.fromisoformat(election_date)
today = datetime.date.today()   # it changes later !!!
sample_n = 1000 # used in statistical error
re_coef = 0.6 # random error coefficient
sample = 2000 # number of simulation
interval_max = 60 # highest gain to calc probability
# source sheet
sheetkey = "1a3i0HfphGxlz-6_E04wYq30tG-gGvWQcyUc4-pNwI9U"
path = "hu-2026/"

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

# --- Seat allocation calculation ---
print("\nCalculating seat allocations...")

# Initialize the seat allocator once for better performance
# Using v2 allocator which implements the exact Google Sheet formulas
# and the Mi Hazánk <5% rule
allocator_v2 = HungarianSeatAllocatorV2()

# Get party columns - need to identify Fidesz, Tisza, Others, and Mi Hazánk
# Assuming the columns are named appropriately in the simulations dataframe
party_columns = simulations_aging_cov.columns.tolist()

# Create a dataframe to store seat allocations for each simulation
# Hungarian parliament has 199 seats total
seats_simulations_aging_cov = pd.DataFrame(0, index=range(sample), columns=['Fidesz', 'Tisza', 'MH'], dtype=int)

# For each simulation, calculate seat allocation
for i in range(sample):
    # Get the vote shares from this simulation (as percentages)
    sim_row = simulations_aging_cov.iloc[i]
    
    # Extract the relevant parties (column names: Fidesz, Tisza, MH, DK, MKKP)
    fidesz_pct = sim_row.get('Fidesz', 0) * 100
    tisza_pct = sim_row.get('Tisza', 0) * 100
    mi_hazank_pct = sim_row.get('MH', 0) * 100
    
    # Calculate "Others" as 100 - Fidesz - Tisza - Mi Hazánk
    others_pct = 100 - fidesz_pct - tisza_pct - mi_hazank_pct
    
    # Allocate seats using the Hungarian v2 function
    # This implements the Mi Hazánk <5% rule: if MH < 5%, it gets 0 list seats
    fidesz_seats, tisza_seats, mi_hazank_seats = allocator_v2.allocate_seats(
        fidesz_pct, tisza_pct, others_pct, mi_hazank_pct
    )
    
    # Store the results
    seats_simulations_aging_cov.iloc[i] = [fidesz_seats, tisza_seats, mi_hazank_seats]

# Calculate probability distribution for each party
# Pr[S=x] for x from 0 to 199 seats
seat_max = 199
seats_prob_aging_cov = pd.DataFrame(index=range(0, seat_max + 1), columns=seats_simulations_aging_cov.columns, dtype=float)

for party in seats_simulations_aging_cov.columns:
    counts = seats_simulations_aging_cov[party].value_counts(sort=False)
    for x in range(0, seat_max + 1):
        seats_prob_aging_cov.loc[x, party] = counts.get(x, 0) / sample

# Write to Google Sheets
wsw = sh.worksheet('seats_aging_cov')
header = ['Pr[S=x]'] + seats_prob_aging_cov.columns.values.tolist()
table = [header] + [[x] + seats_prob_aging_cov.loc[x].values.tolist() for x in seats_prob_aging_cov.index]
wsw.update(values=table, range_name='A1')

print(f"Seat allocation complete. Results written to 'seats_aging_cov' worksheet.")

# --- Seat Rank Calculation (Probability of Winning Most Seats) ---

print("\nCalculating probability of seat rankings (rank 1, 2, 3)...")

# For each simulation, determine the ranking of parties by seats
# In case of a tie, Tisza wins (if Tisza is tied), otherwise random
seats_rank_counts = pd.DataFrame(0, index=seats_simulations_aging_cov.columns, columns=['Rank 1', 'Rank 2', 'Rank 3'], dtype=int)

for i in range(sample):
    sim_seats = seats_simulations_aging_cov.iloc[i]
    
    # Sort parties by seats (descending)
    sorted_parties = sim_seats.sort_values(ascending=False)
    
    # Handle ties with Tisza preference
    # Get unique seat values
    unique_seats = sorted_parties.unique()
    
    # Assign ranks
    rank_assignment = []
    for seat_value in unique_seats:
        tied_parties = sorted_parties[sorted_parties == seat_value].index.tolist()
        
        if len(tied_parties) == 1:
            rank_assignment.append(tied_parties[0])
        else:
            # Tie - Tisza gets priority if present
            if 'Tisza' in tied_parties:
                rank_assignment.append('Tisza')
                tied_parties.remove('Tisza')
            # Add remaining tied parties in original order
            rank_assignment.extend(tied_parties)
    
    # Count ranks (only top 3)
    for rank_idx in range(min(3, len(rank_assignment))):
        party = rank_assignment[rank_idx]
        rank_col = f'Rank {rank_idx + 1}'
        seats_rank_counts.loc[party, rank_col] += 1

# Calculate probabilities
seats_rank_prob = seats_rank_counts / sample

# Write to Google Sheets
wsw = sh.worksheet('seats_rank')
# Format: Party name in column A, probabilities in columns B, C, D
table = [['Party', 'Pr[Rank=1]', 'Pr[Rank=2]', 'Pr[Rank=3]']]
for party in seats_rank_prob.index:
    table.append([
        party, 
        seats_rank_prob.loc[party, 'Rank 1'],
        seats_rank_prob.loc[party, 'Rank 2'],
        seats_rank_prob.loc[party, 'Rank 3']
    ])
wsw.update(values=table, range_name='A1')

print(f"Seat rank probabilities written to 'seats_rank' worksheet.")
print(f"  Fidesz - Rank 1: {seats_rank_prob.loc['Fidesz', 'Rank 1']:.4f}, Rank 2: {seats_rank_prob.loc['Fidesz', 'Rank 2']:.4f}, Rank 3: {seats_rank_prob.loc['Fidesz', 'Rank 3']:.4f}")
print(f"  Tisza  - Rank 1: {seats_rank_prob.loc['Tisza', 'Rank 1']:.4f}, Rank 2: {seats_rank_prob.loc['Tisza', 'Rank 2']:.4f}, Rank 3: {seats_rank_prob.loc['Tisza', 'Rank 3']:.4f}")
print(f"  MH     - Rank 1: {seats_rank_prob.loc['MH', 'Rank 1']:.4f}, Rank 2: {seats_rank_prob.loc['MH', 'Rank 2']:.4f}, Rank 3: {seats_rank_prob.loc['MH', 'Rank 3']:.4f}")

# --- Victory Margin Calculation ---

print("\nCalculating victory margin probabilities...")

try:
    # Get the worksheet for victory margins
    victory_ws = sh.worksheet('victory_margin_aging_cov')
    
    header_row = victory_ws.get('1:1')[0]
    if not header_row or not header_row[0] or 'victory' not in str(header_row[0]).lower():
        print("❌ Error: Could not find expected 'Victory > x' header in cell A1 of victory_margin_aging_cov sheet")
        print("Skipping victory margin calculation")
    else:
        # Get party names from header row (columns B onwards)
        # Filter out 'lo', 'hi' and other non-party columns
        all_headers = [h for h in header_row[1:] if str(h).strip()]
        # Only keep headers that match simulation columns (actual party names)
        sheet_parties = [h for h in all_headers if h in simulations_aging_cov.columns]
        
        if not sheet_parties:
            print("❌ Error: No party headers found in row 1 of victory_margin_aging_cov sheet")
            print(f"   Available simulation parties: {simulations_aging_cov.columns.tolist()}")
            print(f"   Headers in sheet: {all_headers}")
            print("Skipping victory margin calculation")
        else:
            # Read thresholds x from first column (A), starting row 2, until first blank
            first_col = victory_ws.col_values(1)
            thresholds = []
            for v in first_col[1:]:
                if v is None or str(v).strip() == '':
                    break
                try:
                    thresholds.append(float(str(v).replace(',', '.')))
                except ValueError:
                    break
            
            if not thresholds:
                print("❌ Error: No thresholds found in column A (starting A2) of victory_margin_aging_cov sheet")
                print("Skipping victory margin calculation")
            else:
                # Initialize counts dataframe
                victory_margin_counts = pd.DataFrame(0, index=thresholds, columns=sheet_parties, dtype=int)
                
                # For each simulation, calculate victory margin
                for i in range(sample):
                    run = simulations_aging_cov.iloc[i].sort_values(ascending=False)
                    winner_name = run.index[0]
                    winner_score = run.iloc[0]
                    runner_up_score = run.iloc[1]
                    
                    # Victory margin in percentage points
                    margin = (winner_score - runner_up_score) * 100
                    
                    # Count if winner is in our tracked parties
                    if winner_name in victory_margin_counts.columns:
                        for x in thresholds:
                            if margin >= x:
                                victory_margin_counts.loc[x, winner_name] += 1
                
                # Calculate probabilities
                victory_margin_probabilities = victory_margin_counts / sample
                
                # Write to sheet - only update party columns, not the entire table
                # Find the column indices for each party in the header row
                for party in sheet_parties:
                    # Find which column this party is in (use first occurrence)
                    party_col_idx = None
                    for idx, h in enumerate(header_row):
                        if h == party:
                            party_col_idx = idx
                            break  # Use the first occurrence
                    
                    if party_col_idx is not None:
                        # Convert column index to letter (0=A, 1=B, etc.)
                        def num_to_col(num):
                            result = ""
                            while num >= 0:
                                result = chr(65 + (num % 26)) + result
                                num = num // 26 - 1
                            return result
                        
                        col_letter = num_to_col(party_col_idx)
                        # Write only this party's column data (starting from row 2)
                        # Convert the entire column to a list to avoid Series issues
                        party_values = victory_margin_probabilities.loc[:, party].values.tolist()
                        
                        # Debug output
                        print(f"  Writing {party} to column {col_letter}")
                        print(f"  First 3 values: {party_values[:3]}")
                        
                        # Handle nested lists - if values are lists, take the first element
                        if party_values and isinstance(party_values[0], list):
                            print(f"  Detected nested lists, flattening...")
                            party_values = [v[0] if isinstance(v, list) else v for v in party_values]
                        
                        # Ensure we have simple float values
                        col_data = [[float(val)] for val in party_values]
                        range_name = f"{col_letter}2:{col_letter}{1 + len(thresholds)}"
                        
                        print(f"  Range: {range_name}, writing {len(col_data)} values")
                        
                        victory_ws.update(values=col_data, range_name=range_name)
                
                print(f"✓ Victory margin probabilities written successfully for parties: {sheet_parties}")

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
