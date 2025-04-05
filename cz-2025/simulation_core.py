# simulation_core.py
# Contains the core simulation engine functions, independent of data source.

import pandas as pd
import numpy as np
import math
from numpy.random import multivariate_normal
from scipy.linalg import LinAlgError
from itertools import combinations
from typing import Tuple, Dict, List, Optional, Union, Callable

def generate_poll_simulations(
  mu_latest: pd.Series,
  sigman_latest: pd.Series,
  corr_matrix: pd.DataFrame,
  num_runs: int,
  sample_n: int,
  error_coef: float = 2.0
) -> pd.DataFrame:
  """
  Generates simulated poll results using multivariate normal distribution.
  (Same implementation as before)
  """
  # --- Input Validation ---
  if mu_latest.empty or sigman_latest.empty or corr_matrix.empty:
    raise ValueError("Input mu_latest, sigman_latest, and corr_matrix cannot be empty.")
  if not mu_latest.index.equals(sigman_latest.index):
    raise ValueError("Index mismatch between mu_latest and sigman_latest.")
  if not mu_latest.index.equals(corr_matrix.index):
    raise ValueError("Index mismatch between mu_latest and corr_matrix index.")
  if not mu_latest.index.equals(corr_matrix.columns):
    raise ValueError("Index mismatch between mu_latest and corr_matrix columns.")
  if sample_n <= 0:
    raise ValueError("sample_n must be positive.")
  if num_runs <= 0:
    raise ValueError("num_runs must be positive.")

  print(f"Generating {num_runs} poll simulations...")
  print(f"Using error coefficient: {error_coef}")

  # --- Prepare inputs ---
  parties = mu_latest.index
  mu_aligned = mu_latest.loc[parties].fillna(0) # Fill NaNs in mean with 0
  sigman_aligned = sigman_latest.loc[parties].fillna(0) # Fill NaNs in sigma with 0
  corr_aligned = corr_matrix.loc[parties, parties].fillna(0) # Fill NaNs in corr with 0

  # --- Construct Covariance Matrix ---
  diagonal_matrix = np.diag(sigman_aligned.to_numpy() * error_coef)
  corr_np = corr_aligned.to_numpy()
  covariance_matrix = np.matmul(np.matmul(diagonal_matrix, corr_np), diagonal_matrix)

  # --- Check Positive Semi-Definite ---
  try:
    eigenvalues = np.linalg.eigvalsh(covariance_matrix)
    if np.any(eigenvalues < -1e-9):
      print(f"Warning: Covariance matrix has negative eigenvalues: {eigenvalues[eigenvalues < 0]}")
      # Consider adding a small nudge if this becomes problematic
      # covariance_matrix += np.eye(len(parties)) * 1e-9
  except LinAlgError:
    print("Warning: LinAlgError checking eigenvalues. Matrix might be singular.")

  # --- Generate Samples ---
  mean_vector = mu_aligned.to_numpy() * sample_n
  print("Running multivariate normal sampling...")
  try:
    raw_samples = multivariate_normal(mean=mean_vector, cov=covariance_matrix, size=num_runs)
  except LinAlgError as e:
    print("\n>>> ERROR: Covariance matrix is not positive semi-definite.")
    print(f"Original error: {e}")
    # Add more debug info if needed:
    # print("Mean Vector:", mean_vector)
    # print("Covariance Matrix:\n", covariance_matrix)
    # print("Correlation Matrix:\n", corr_aligned)
    # print("Sigman Aligned:\n", sigman_aligned)
    raise

  # --- Format Output ---
  simulated_polls_df = pd.DataFrame(raw_samples / sample_n, columns=parties)
  simulated_polls_df = simulated_polls_df.clip(0, 1) # Clip to valid percentage range

  print(f"Generated {len(simulated_polls_df)} simulation runs.")
  return simulated_polls_df


def calculate_seats_imperiali(
  poll_sample_series: pd.Series,
  regional_results_df: pd.DataFrame,
  regions_seats_df: pd.DataFrame,
  total_last_votes: float
) -> pd.Series:
  """
  Calculates parliamentary seats based on a single poll sample using the
  Imperiali quota method with two scrutinies.
  (Same implementation as before)
  """
  # --- Prepare input ---
  if poll_sample_series.name is None:
      poll_sample_series.name = 'poll_value'
  sample_df = poll_sample_series.reset_index().rename(columns={'index': 'party'})

  # Ensure 'needs' column exists in regional_results_df
  if 'needs' not in regional_results_df.columns:
      raise ValueError("'needs' column (threshold) missing from regional_results_df.")

  regional_sample = regional_results_df.merge(sample_df, on="party", how="left")
  regional_sample['poll_value'] = regional_sample['poll_value'].fillna(0.0)
  regional_sample['estimated_votes'] = regional_sample['poll_value'] * regional_sample['rate'] * total_last_votes

  # --- National Threshold Check ---
  national_totals = regional_sample.groupby('party').agg(
      estimated_votes_total=('estimated_votes', 'sum'),
      needs=('needs', 'first')
  ).reset_index()
  total_estimated_votes_all_parties = national_totals['estimated_votes_total'].sum()
  if total_estimated_votes_all_parties > 0:
      national_totals['national_share'] = national_totals['estimated_votes_total'] / total_estimated_votes_all_parties
  else:
      national_totals['national_share'] = 0.0
  parties_over_threshold = national_totals[national_totals['national_share'] >= national_totals['needs']]['party'].tolist()

  regional_sample_filtered = regional_sample[regional_sample['party'].isin(parties_over_threshold)].copy()
  if regional_sample_filtered.empty:
      return pd.Series(0, index=sample_df['party'].unique(), name='seats')

  # --- 1st Scrutinium ---
  regional_seats_list = []
  region_codes = regions_seats_df['region_code'].unique()
  for rc in region_codes:
      region_data = regional_sample_filtered[regional_sample_filtered['region_code'] == rc].reset_index()
      if region_data.empty: continue
      s = region_data['estimated_votes'].sum()
      try:
          rs = regions_seats_df.loc[regions_seats_df['region_code'] == rc, 'seats'].iloc[0]
      except IndexError:
          print(f"Warning: Region code {rc} not found in regions_seats_df. Skipping region.")
          continue
      if s == 0 or rs == 0: continue

      # Imperiali quota N
      N_float = s / (rs + 2) # Keep as float for calculations
      # N = round(N_float) # Round only if strictly necessary by law? Let's use float.

      if N_float == 0: continue

      region_data['nof_seats'] = (region_data['estimated_votes'] / N_float).apply(math.floor)
      region_data['rest'] = region_data['estimated_votes'] - region_data['nof_seats'] * N_float # Use float N

      # Correction from original code (if total floor seats > available seats)
      overseats = region_data['nof_seats'].sum() - rs
      if overseats > 0:
          seat_winners = region_data[region_data['nof_seats'] > 0].copy()
          if not seat_winners.empty:
              seat_winners['rest_rank_asc'] = seat_winners['rest'].rank(method='first', ascending=True)
              indices_to_reduce = seat_winners[seat_winners['rest_rank_asc'] <= overseats].index
              region_data.loc[indices_to_reduce, 'nof_seats'] -= 1
              # Recalculate rests only for those reduced
              region_data.loc[indices_to_reduce, 'rest'] = region_data.loc[indices_to_reduce, 'estimated_votes'] - region_data.loc[indices_to_reduce, 'nof_seats'] * N_float

      regional_seats_list.append(region_data[['party', 'region_code', 'nof_seats', 'rest']])

  if not regional_seats_list:
      return pd.Series(0, index=sample_df['party'].unique(), name='seats')
  regional_seats_df_combined = pd.concat(regional_seats_list, ignore_index=True)

  # --- 2nd Scrutinium ---
  total_seats_1st = regional_seats_df_combined['nof_seats'].sum()
  seats_remaining_2nd = 200 - total_seats_1st
  if seats_remaining_2nd <= 0:
      final_seats = regional_seats_df_combined.groupby('party')['nof_seats'].sum().astype(int)
  else:
      total_rests = regional_seats_df_combined['rest'].sum()
      if total_rests <= 0: # Use <= to handle potential float precision issues
          final_seats = regional_seats_df_combined.groupby('party')['nof_seats'].sum().astype(int)
      else:
          RN = total_rests / (seats_remaining_2nd + 1)
          if RN <= 0:
               print("Warning: Republic Number (RN) is zero or negative in 2nd scrutiny.")
               extras_seats_2nd = pd.DataFrame({'party': regional_sample_filtered['party'].unique(), 'extra': 0})
               extras_seats_2nd = extras_seats_2nd.drop_duplicates() # Ensure unique parties
          else:
              party_rests = regional_seats_df_combined.groupby('party')['rest'].sum().reset_index()
              party_rests['extra_seats_floor'] = (party_rests['rest'] / RN).apply(math.floor)
              party_rests['rest_rest'] = party_rests['rest'] - party_rests['extra_seats_floor'] * RN
              party_rests['rest_rest_rank'] = party_rests['rest_rest'].rank(method='first', ascending=False)
              seats_allocated_floor = party_rests['extra_seats_floor'].sum()
              seats_left_after_floor = seats_remaining_2nd - seats_allocated_floor
              if seats_left_after_floor < 0: seats_left_after_floor = 0
              party_rests['extra_rank_seats'] = (party_rests['rest_rest_rank'] <= seats_left_after_floor).astype(int)
              party_rests['extra'] = party_rests['extra_seats_floor'] + party_rests['extra_rank_seats']
              extras_seats_2nd = party_rests[['party', 'extra']]

          seats_1st = regional_seats_df_combined.groupby('party')['nof_seats'].sum().reset_index()
          final_seats_calc = seats_1st.merge(extras_seats_2nd, on='party', how='outer') # Use outer merge
          final_seats_calc['nof_seats'] = final_seats_calc['nof_seats'].fillna(0) # Fill seats for parties only in 2nd scrutiny
          final_seats_calc['extra'] = final_seats_calc['extra'].fillna(0)
          final_seats_calc['seats'] = final_seats_calc['nof_seats'] + final_seats_calc['extra']
          final_seats = final_seats_calc.set_index('party')['seats'].astype(int)

  # --- Format Output ---
  all_parties_final = sample_df['party'].unique()
  final_seats = final_seats.reindex(all_parties_final, fill_value=0)
  if abs(final_seats.sum() - 200) > 1e-6 : # Check for deviation
      print(f"Warning: Total allocated seats ({final_seats.sum()}) is not 200.")
  return final_seats.rename('seats')


def run_seat_simulations(
  simulated_polls_df: pd.DataFrame,
  seat_calc_function: Callable, # Use Callable type hint
  seat_calc_args: dict
) -> pd.DataFrame:
  """
  Runs the seat calculation function for each simulated poll outcome.
  (Same implementation as before)
  """
  if simulated_polls_df.empty:
    print("Warning: Input simulated_polls_df is empty. Returning empty DataFrame.")
    return pd.DataFrame()

  print(f"Running seat simulations for {len(simulated_polls_df)} runs...")
  all_seat_results = []
  total_runs = len(simulated_polls_df)
  report_interval = max(1, total_runs // 10)

  for i, (index, poll_sample_series) in enumerate(simulated_polls_df.iterrows()):
    if (i + 1) % report_interval == 0 or i == total_runs - 1:
      print(f"  Calculating seats for simulation {i+1}/{total_runs}...")

    # Ensure series has a name (required by calculate_seats_imperiali)
    poll_sample_series.name = 'poll_value'

    try:
      seats_series = seat_calc_function(
          poll_sample_series=poll_sample_series,
          **seat_calc_args
      )
      all_seat_results.append(seats_series)
    except Exception as e:
      print(f"\nError calculating seats for simulation index {index}: {e}")
      # Optionally append NaNs or skip - let's skip for now
      continue

  if not all_seat_results:
    print("Warning: No seat calculation results were generated.")
    return pd.DataFrame(columns=simulated_polls_df.columns)

  # Concatenate results and ensure correct structure
  simulated_seats_df = pd.concat(all_seat_results, axis=1).T
  simulated_seats_df.index = simulated_polls_df.index[:len(simulated_seats_df)] # Realign index if runs were skipped
  simulated_seats_df = simulated_seats_df.fillna(0).astype(int) # Fill NaNs and ensure integer seats

  print("Seat simulations finished.")
  return simulated_seats_df


def calculate_simulation_stats(
  simulated_seats_df: pd.DataFrame,
  best_estimate_seats_series: pd.Series,
  choices_df: pd.DataFrame
) -> pd.DataFrame:
  """
  Calculates summary statistics from simulated seat results.
  (Same implementation as before, minor cleanup)
  """
  if simulated_seats_df.empty:
    print("Warning: simulated_seats_df is empty. Cannot calculate stats.")
    return pd.DataFrame()

  print("Calculating statistics from seat simulations...")
  num_runs = len(simulated_seats_df)

  # Calculate basic stats
  stats_calc = pd.DataFrame({
    'party': simulated_seats_df.columns,
    'median': simulated_seats_df.median(axis=0).values,
    'lo': simulated_seats_df.quantile(q=0.05, axis=0, interpolation='nearest').values,
    'hi': simulated_seats_df.quantile(q=0.95, axis=0, interpolation='nearest').values,
    'in': (simulated_seats_df > 0).sum(axis=0).values / num_runs
  })

  # Prepare best estimate seats for merge
  best_estimate_df = best_estimate_seats_series.reset_index()
  best_estimate_df.columns = ['party', 'seats'] # Ensure column names are correct

  # Merge best estimate
  stats = stats_calc.merge(best_estimate_df, on='party', how='left')
  stats['seats'] = stats['seats'].fillna(0).astype(int)

  # Merge with choices data
  required_choice_cols = ['id', 'abbreviation', 'color', 'logo', 'mps']
  available_choice_cols = [col for col in required_choice_cols if col in choices_df.columns]
  if len(available_choice_cols) < len(required_choice_cols):
      print(f"Warning: choices_df missing columns: {set(required_choice_cols) - set(available_choice_cols)}")

  choices_subset = choices_df[available_choice_cols].copy()
  stats = stats.merge(choices_subset, left_on='party', right_on='id', how='left')

  # Calculate difference
  if 'mps' in stats.columns:
    stats['mps'] = stats['mps'].fillna(0).astype(int)
    stats['difference'] = stats['seats'] - stats['mps']
  else:
    stats['difference'] = np.nan

  # Sort results
  stats = stats.sort_values(['seats', 'hi', 'median', 'lo'], ascending=[False, False, False, False])

  # Select and reorder final columns
  final_cols_order = [
    'party', 'abbreviation', 'seats', 'median', 'lo', 'hi', 'in',
    'difference', 'mps', 'color', 'logo', 'id'
  ]
  final_cols_exist = [col for col in final_cols_order if col in stats.columns]
  stats_final = stats[final_cols_exist].reset_index(drop=True) # Reset index

  print("Simulation statistics calculation complete.")
  return stats_final


def calculate_coalition_probabilities(
  simulated_seats_df: pd.DataFrame,
  stats_df: pd.DataFrame,
  majority_threshold: int,
  predefined_coalitions_list: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """
  Calculates majority probabilities for single parties, pairs, and predefined coalitions.
  (Same implementation as before)
  """
  if simulated_seats_df.empty:
    print("Warning: simulated_seats_df is empty. Cannot calculate coalition probabilities.")
    return pd.DataFrame(), pd.DataFrame()
  if 'party' not in stats_df.columns or 'seats' not in stats_df.columns:
    raise ValueError("stats_df must contain 'party' and 'seats' columns.")

  print(f"Calculating coalition probabilities (Majority threshold: {majority_threshold})...")
  num_runs = len(simulated_seats_df)
  parties = simulated_seats_df.columns.tolist()
  # Use stats_df directly for lookup, assuming 'party' is unique
  stats_map = stats_df.set_index('party')['seats']

  results_exclusive = []
  results_inclusive = []

  # --- Helper ---
  def _get_probabilities(coalition_parties: List[str]) -> Tuple[float, float]:
    valid_parties = [p for p in coalition_parties if p in simulated_seats_df.columns]
    if not valid_parties: return 0.0, 0.0

    coalition_total_seats_all_runs = simulated_seats_df[valid_parties].sum(axis=1)
    prob_inclusive = (coalition_total_seats_all_runs >= majority_threshold).mean()

    condition = (simulated_seats_df[valid_parties] > 0).all(axis=1)
    simulated_seats_filtered = simulated_seats_df.loc[condition]
    if simulated_seats_filtered.empty:
      prob_exclusive = 0.0
    else:
      coalition_total_seats_filtered = simulated_seats_filtered[valid_parties].sum(axis=1)
      prob_exclusive = (coalition_total_seats_filtered >= majority_threshold).mean()
    return prob_exclusive, prob_inclusive

  # --- Singles ---
  print("  Processing single parties...")
  for party in parties:
    prob_ex, prob_in = _get_probabilities([party])
    best_seats = stats_map.get(party, 0)
    results_exclusive.append({'id': party, 'majority_probability': prob_ex, 'seats': best_seats})
    results_inclusive.append({'id': party, 'majority_probability': prob_in, 'seats': best_seats})

  # --- Pairs ---
  print("  Processing pairs of parties...")
  for p1, p2 in combinations(parties, 2):
    prob_ex, prob_in = _get_probabilities([p1, p2])
    best_seats_p1 = stats_map.get(p1, 0); best_seats_p2 = stats_map.get(p2, 0)
    total_best_seats = best_seats_p1 + best_seats_p2
    coalition_id = f"{p2}*{p1}" if best_seats_p1 < best_seats_p2 else f"{p1}*{p2}"
    results_exclusive.append({'id': coalition_id, 'majority_probability': prob_ex, 'seats': total_best_seats})
    results_inclusive.append({'id': coalition_id, 'majority_probability': prob_in, 'seats': total_best_seats})

  # --- Predefined ---
  if predefined_coalitions_list:
    print("  Processing predefined coalitions...")
    for coalition_str in predefined_coalitions_list:
      coalition_parties = coalition_str.split('*')
      valid_parties_in_coalition = [p for p in coalition_parties if p in stats_map.index]
      if len(valid_parties_in_coalition) != len(coalition_parties):
        print(f"    Warning: Skipping predefined coalition '{coalition_str}' due to unknown member parties: {set(coalition_parties) - set(valid_parties_in_coalition)}")
        continue
      if not valid_parties_in_coalition: continue

      prob_ex, prob_in = _get_probabilities(valid_parties_in_coalition)
      total_best_seats = sum(stats_map.get(p, 0) for p in valid_parties_in_coalition)
      results_exclusive.append({'id': coalition_str, 'majority_probability': prob_ex, 'seats': total_best_seats})
      results_inclusive.append({'id': coalition_str, 'majority_probability': prob_in, 'seats': total_best_seats})

  # --- Create and Sort DataFrames ---
  coalitions_exclusive_df = pd.DataFrame(results_exclusive)
  coalitions_inclusive_df = pd.DataFrame(results_inclusive)
  if not coalitions_exclusive_df.empty:
    coalitions_exclusive_df = coalitions_exclusive_df.sort_values(['majority_probability', 'seats'], ascending=[False, False]).reset_index(drop=True)
  if not coalitions_inclusive_df.empty:
    coalitions_inclusive_df = coalitions_inclusive_df.sort_values(['majority_probability', 'seats'], ascending=[False, False]).reset_index(drop=True)

  print("Coalition probability calculation finished.")
  return coalitions_exclusive_df, coalitions_inclusive_df