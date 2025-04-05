# run_manual_analysis.py
# Runner script for Project 2: Analysis based on manual GSheet data.

# %%
# --- Setup Python Path ---
import sys
import os
import importlib
from numpy.linalg import LinAlgError # Import error type
import datetime # Import datetime

# %%
# --- RELOAD CUSTOM MODULES ---
try:
  if 'simulation_core' in locals() or 'simulation_core' in globals():
    print(f"Reloading simulation_core...")
    simulation_core = importlib.reload(simulation_core)
  if 'manual_data_preparer' in locals() or 'manual_data_preparer' in globals():
    print(f"Reloading manual_data_preparer...")
    manual_data_preparer = importlib.reload(manual_data_preparer)
  print("Reload complete.")
except NameError:
  print("Modules not imported yet, skipping reload.")
except Exception as e:
  print(f"Error reloading modules: {e}")


# %%
# --- Setup, Imports ---
try:
  script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
  script_dir = os.getcwd()

if script_dir not in sys.path: sys.path.insert(0, script_dir)

import pandas as pd
import numpy as np
import math
import gspread # Import gspread for writing results
# Import the refactored modules
import simulation_core
import manual_data_preparer

try:
  print(f"Imported simulation_core from: {simulation_core.__file__}")
  print(f"Imported manual_data_preparer from: {manual_data_preparer.__file__}")
except Exception as e:
  print(f"ERROR: Failed to import custom modules: {e}")
  sys.exit(1)


# %%
# --- Configuration for Manual Analysis ---
print("Setting configuration for Manual Analysis...")
# Google Sheet Info
SHEET_KEY = "1es2J0O_Ig7RfnVHG3bHmX8SBjlMvrPwn4s1imYkxbwg"
PREF_WORKSHEET = 'preference'
CORR_WORKSHEET = 'median correlations'
CUSTOM_COALITIONS_WS = 'vlastní_koalice'

# Output Worksheets (matching original script)
RANKS_AGING_COV_WS = 'pořadí_aktuální_aging_cov'
PROBS_AGING_COV_WS = 'pravděpodobnosti_aktuální_aging_cov'
DUELS_AGING_COV_WS = 'duely_aging_cov'
TOP2_COV_WS = 'top_2_cov'
NUM_IN_AGING_COV_WS = 'number_in_aging_cov'
PREF_WRITEBACK_WS = 'preference' # For timestamp
HISTORY_WRITEBACK_WS = 'history' # For archiving preferences
COALITIONS_EXCL_WS = 'koalice_excl' # 
COALITIONS_INCL_WS = 'koalice_inc' # 
PROB_IN_WS = 'in' #

# Paths for Auxiliary Data (Seat Calculation - same as Project 1)
# *** ADJUST THIS PATH ***
DATA_PATH_PREFIX = "./data/" # Assuming data folder is sibling to script dir
REGIONAL_RESULTS_CSV = DATA_PATH_PREFIX + "psp2021_regional_results.csv"
REGIONS_SEATS_CSV = DATA_PATH_PREFIX + "psp2021_seats.csv"
CHOICES_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRhp47e91OazMSiu56gOTsUtnFEIaJiIhJbsgNTylwt89XIEnbiVyObJ8xHEoZPObo6ntOmQ9Tg-sf9/pub?gid=302501468&single=true&output=csv" # For choices metadata

# Simulation Parameters
SAMPLE_N_MANUAL = 1000 # sample_n from original script
NUM_RUNS_MANUAL = 2000 # sample from original script
# Note: Original script applied 0.9 factor during error calculation AND added uniform noise.
# The 'error_coef' here scales the covariance matrix for the NORMAL part.
# Let's set it to 1 and let manual_data_preparer handle volatility scaling,
# and add uniform noise separately. Adjust if needed.
ERROR_COEF_MANUAL = 1.0
# Coefficient for the UNIFORM noise component (based on 1.5 * 0.9 * 0.9 from original script notes)
UNIFORM_NOISE_COEF = 1.215

# Aging Parameters
ELECTION_DATE_MANUAL = '2025-09-29' # From original script
AGING_POWER_MANUAL = 1.15

# Seat Calculation Parameters (same as Project 1)
TOTAL_LAST_VOTES = 5375090
MAJORITY_THRESHOLD = 101

# Interval Calculation Parameters
INTERVAL_MAX = 60
ADDITIONAL_POINTS = [] # Add specific points if needed, e.g., [5.0]

# Control Flags
USE_AGING = True # Use the aged sigman for simulation
ADD_UNIFORM_NOISE = True # Add the extra uniform noise step
LOAD_CUSTOM_COALITIONS = True # Load custom coalitions from sheet

print("Configuration set.")


# %%
# --- Load Auxiliary Seat Calc Data ---
print("\n--- Loading Auxiliary Seat Calculation Data ---")
last_regional_results = pd.DataFrame()
regions_seats = pd.DataFrame()
choices_data_aux = pd.DataFrame() # For metadata (mps, abbreviation)
try:
  last_regional_results_raw = pd.read_csv(REGIONAL_RESULTS_CSV)
  regions_seats = pd.read_csv(REGIONS_SEATS_CSV)
  choices_data_aux = pd.read_csv(CHOICES_URL) # Need this for stats_df later

  # Merge 'needs' from manual sheet later, this is just base data
  # Assuming 'party', 'rate', 'votes' exist in regional results
  required_regional_cols = ['party', 'region_code', 'votes', 'rate']
  if not all(col in last_regional_results_raw.columns for col in required_regional_cols):
    raise ValueError(f"Regional results CSV missing required columns: {required_regional_cols}")
  last_regional_results = last_regional_results_raw[required_regional_cols].copy() # Keep only needed cols

  if 'region_code' not in regions_seats.columns or 'seats' not in regions_seats.columns:
    raise ValueError("Region seats CSV missing region_code or seats column.")
  
  # --- Load custom coalitions --- # <-- NEW Section
  if LOAD_CUSTOM_COALITIONS:
    custom_coalitions_list = manual_data_preparer.load_custom_coalitions_from_sheet(
      sheetkey=SHEET_KEY,
      worksheet_name=CUSTOM_COALITIONS_WS
    )
  else:
    print("Skipping custom coalitions loading.")

  print("Auxiliary seat calculation data loaded.")
except Exception as e:
  print(f"\n>>> Error loading auxiliary seat calc data: {e}")
  # Ensure variables are defined
  last_regional_results = pd.DataFrame()
  regions_seats = pd.DataFrame()
  choices_data_aux = pd.DataFrame()


# %%
# --- Step 1: Load Manual Data & Prepare Inputs ---
print("\n--- 1. Loading Manual Data & Preparing Inputs ---")
preference_df = pd.DataFrame()
sheet_date = None
mu_latest = pd.Series(dtype=float)
sigman_latest = pd.Series(dtype=float)
sigman_latest_aged = pd.Series(dtype=float)
corr_matrix = pd.DataFrame()
try:
  # Load preferences
  preference_df, sheet_date = manual_data_preparer.load_manual_poll_data(
    sheetkey=SHEET_KEY, worksheet_name=PREF_WORKSHEET
  )
  mu_latest = preference_df.set_index('party')['p'] # Mean is just the preference 'p'

  # Calculate Sigman (and aged version)
  sigman_latest, sigman_latest_aged = manual_data_preparer.calculate_manual_sigman(
    preference_df=preference_df,
    sample_n=SAMPLE_N_MANUAL,
    sheet_date=sheet_date if USE_AGING else None, # Pass dates only if aging
    election_date=ELECTION_DATE_MANUAL if USE_AGING else None,
    aging_power=AGING_POWER_MANUAL
  )

  # Load Correlation
  corr_matrix = manual_data_preparer.load_manual_correlation(
    sheetkey=SHEET_KEY,
    party_order=mu_latest.index.tolist(), # Use order from preferences
    worksheet_name=CORR_WORKSHEET
  )

  # --- Choose Sigman for Simulation ---
  if USE_AGING and sigman_latest_aged is not None:
    sigman_to_use = sigman_latest_aged
    aging_factor_for_noise = manual_data_preparer.calculate_aging_coeff(
      sheet_date, datetime.date.fromisoformat(ELECTION_DATE_MANUAL), AGING_POWER_MANUAL
    )
    print("Using AGED sigman for simulation.")
  else:
    sigman_to_use = sigman_latest
    aging_factor_for_noise = 1.0
    if USE_AGING: print("Warning: Aging requested but failed. Using non-aged sigman.")
    else: print("Using NON-AGED sigman for simulation.")

  # --- Align Inputs ---
  common_index = mu_latest.index.intersection(sigman_to_use.index).intersection(corr_matrix.index)
  if len(common_index) < len(mu_latest.index):
    print(f"Warning: Aligning inputs. Parties dropped: {set(mu_latest.index) - set(common_index)}")

  mu_latest_aligned = mu_latest.loc[common_index]
  sigman_latest_aligned = sigman_to_use.loc[common_index]
  corr_matrix_aligned = corr_matrix.loc[common_index, common_index]

  if mu_latest_aligned.empty or sigman_latest_aligned.empty or corr_matrix_aligned.empty:
    raise ValueError("Inputs for simulation are empty after alignment.")

  print("Manual data loaded and inputs prepared.")

except Exception as e:
  print(f"\n>>> Error preparing manual inputs: {e}")
  # Ensure variables are defined
  preference_df = pd.DataFrame()
  mu_latest_aligned = pd.Series(dtype=float)
  sigman_latest_aligned = pd.Series(dtype=float)
  corr_matrix_aligned = pd.DataFrame()


# %%
# --- Step 2: Generate Base Poll Simulations (CORE) ---
print("\n--- 2. Generating Base Poll Simulations (Multivariate Normal) ---")
simulated_polls_mvn = pd.DataFrame()
if not mu_latest_aligned.empty:
  try:
    simulated_polls_mvn = simulation_core.generate_poll_simulations(
      mu_latest=mu_latest_aligned,
      sigman_latest=sigman_latest_aligned,
      corr_matrix=corr_matrix_aligned,
      num_runs=NUM_RUNS_MANUAL,
      sample_n=SAMPLE_N_MANUAL, # Note: passed sample_n used for scaling mean/cov
      error_coef=ERROR_COEF_MANUAL # Scales sigman in cov matrix build
    )
    print(f"Generated MVN simulations. Shape: {simulated_polls_mvn.shape}")
  except (ValueError, LinAlgError, Exception) as e:
    print(f"\n>>> Error generating MVN simulations: {e}")
    simulated_polls_mvn = pd.DataFrame()
else:
  print("Skipping MVN simulation: Input preparation failed.")


# %%
# --- Step 3: Add Uniform Noise (Manual Project Specific) ---
print("\n--- 3. Adding Uniform Noise ---")
simulated_polls_final = pd.DataFrame()
if not simulated_polls_mvn.empty and ADD_UNIFORM_NOISE:
  try:
    simulated_polls_final = manual_data_preparer.add_uniform_noise(
      simulated_polls_mvn=simulated_polls_mvn,
      preference_df=preference_df, # Contains volatility, p
      sample_n=SAMPLE_N_MANUAL, # Base N for sdx calc
      aging_factor=aging_factor_for_noise, # Use calculated aging factor
      uniform_error_coef=UNIFORM_NOISE_COEF
    )
    print(f"Added uniform noise. Final simulations shape: {simulated_polls_final.shape}")
  except Exception as e:
    print(f"\n>>> Error adding uniform noise: {e}")
    simulated_polls_final = simulated_polls_mvn # Fallback to MVN results if noise fails? Or empty?
elif not ADD_UNIFORM_NOISE:
  simulated_polls_final = simulated_polls_mvn # Use MVN results if noise is skipped
  print("Skipping uniform noise as per configuration.")
else:
  print("Skipping uniform noise: Base MVN simulation failed.")


# %%
# --- Step 4: Calculate Best Estimate Seats (CORE) ---
# Note: Uses the base mu_latest (without noise) for the single best estimate
print("\n--- 4. Calculating Best Estimate Seats ---")
best_estimate_seats = pd.Series(dtype=int)
# Merge manual 'needs' into regional results for seat calculation
if not mu_latest_aligned.empty and not last_regional_results.empty and not regions_seats.empty and not preference_df.empty:
  try:
    # Prepare regional results with manual needs
    needs_map = preference_df.set_index('party')['needed']
    regional_results_with_needs = last_regional_results.copy()
    regional_results_with_needs['needs'] = regional_results_with_needs['party'].map(needs_map)
    # Handle parties in regional results but not in manual preferences (fallback need?)
    if regional_results_with_needs['needs'].isna().any():
      missing_need_parties = regional_results_with_needs[regional_results_with_needs['needs'].isna()]['party'].unique()
      print(f"Warning: Missing 'needed' threshold for {missing_need_parties} in best estimate calc. Using 0.05.")
      regional_results_with_needs['needs'] = regional_results_with_needs['needs'].fillna(0.05)

    # Align mu_latest with parties present in regional data
    mu_for_best = mu_latest_aligned.copy()
    common_parties_best = mu_for_best.index.intersection(regional_results_with_needs['party'].unique())
    mu_for_best_aligned = mu_for_best.loc[common_parties_best]
    mu_for_best_aligned.name = 'poll_value'

    if not mu_for_best_aligned.empty:
      seat_calc_args_best = {
        'regional_results_df': regional_results_with_needs, # Use version with manual needs
        'regions_seats_df': regions_seats,
        'total_last_votes': TOTAL_LAST_VOTES
      }
      best_estimate_seats = simulation_core.calculate_seats_imperiali(
        poll_sample_series=mu_for_best_aligned, **seat_calc_args_best
      )
      print("Calculated best estimate seats.")
    else: print("Skipping: No common parties between mu and regional data.")

  except Exception as e:
    print(f"\n>>> Error calculating best estimate seats: {e}")
    best_estimate_seats = pd.Series(dtype=int)
else: print("Skipping: Previous steps failed or auxiliary data missing.")


# %%
# --- Step 5: Run Seat Simulations (CORE) ---
# Uses the FINAL simulated polls (potentially with noise)
print("\n--- 5. Running Seat Simulations ---")
simulated_seats = pd.DataFrame()
if not simulated_polls_final.empty and not last_regional_results.empty and not regions_seats.empty and not preference_df.empty:
  try:
    # Prepare regional results with manual needs (as in step 4)
    needs_map = preference_df.set_index('party')['needed']
    regional_results_with_needs = last_regional_results.copy()
    regional_results_with_needs['needs'] = regional_results_with_needs['party'].map(needs_map)
    if regional_results_with_needs['needs'].isna().any(): # Handle missing needs again
      regional_results_with_needs['needs'] = regional_results_with_needs['needs'].fillna(0.05)

    seat_calc_args_sim = {
      'regional_results_df': regional_results_with_needs, # Use version with manual needs
      'regions_seats_df': regions_seats,
      'total_last_votes': TOTAL_LAST_VOTES
    }
    simulated_seats = simulation_core.run_seat_simulations(
      simulated_polls_df=simulated_polls_final, # Use final simulations
      seat_calc_function=simulation_core.calculate_seats_imperiali,
      seat_calc_args=seat_calc_args_sim
    )
    print(f"Generated simulated seats. Shape: {simulated_seats.shape}")
  except Exception as e:
    print(f"\n>>> Error running seat simulations: {e}")
    simulated_seats = pd.DataFrame()
else: print("Skipping: Previous steps failed or required data missing.")


# %%
# --- Step 6: Calculate Simulation Statistics (CORE) ---
print("\n--- 6. Calculating Core Simulation Statistics ---")
stats_df = pd.DataFrame()
if not simulated_seats.empty and not best_estimate_seats.empty and not choices_data_aux.empty:
  try:
    # The core stats function needs 'mps', 'abbreviation' etc. from choices_data_aux
    stats_df = simulation_core.calculate_simulation_stats(
      simulated_seats_df=simulated_seats,
      best_estimate_seats_series=best_estimate_seats,
      choices_df=choices_data_aux # Use standard choices data for metadata
    )
    print(f"Calculated core statistics. Shape: {stats_df.shape}")
    # print(stats_df.head())
  except Exception as e:
    print(f"\n>>> Error calculating core simulation statistics: {e}")
    stats_df = pd.DataFrame()
else: print("Skipping: Previous steps failed or required data missing.")


# %%
# --- Step 7: Calculate Coalition Probabilities (CORE) ---
# Note: Predefined coalitions are not typically used in this manual analysis, but the function handles it
print("\n--- 7. Calculating Coalition Probabilities ---")
coalitions_exclusive = pd.DataFrame()
coalitions_inclusive = pd.DataFrame()
if not simulated_seats.empty and not stats_df.empty:
  try:
    coalitions_exclusive, coalitions_inclusive = simulation_core.calculate_coalition_probabilities(
      simulated_seats_df=simulated_seats,
      stats_df=stats_df,
      majority_threshold=MAJORITY_THRESHOLD,
      # Use the list loaded from the custom sheet
      predefined_coalitions_list=custom_coalitions_list # Pass None if not loading predefined
    )
    print("Calculated coalition probabilities.")
    # print("\nExclusive (Top 5):\n", coalitions_exclusive.head())
    # print("\nInclusive (Top 5):\n", coalitions_inclusive.head())
  except Exception as e:
    print(f"\n>>> Error calculating coalition probabilities: {e}")
    coalitions_exclusive = pd.DataFrame()
    coalitions_inclusive = pd.DataFrame()
else: print("Skipping: Previous steps failed or required data missing.")


# %%
# --- Step 8: Calculate Manual Project Specific Statistics ---
# (Rankings, Intervals, Duels, Top2, Number In)
# These use simulated_polls_final (with noise, aged)
print("\n--- 8. Calculating Manual Project Specific Statistics ---")

ranks_stats_final = pd.DataFrame()
probs_stats_final = pd.DataFrame()
probs_interval_final = pd.DataFrame()
duels_stats_final = pd.DataFrame()
top2_stats_final = pd.DataFrame()
num_in_stats_final = pd.DataFrame()

if not simulated_polls_final.empty and not preference_df.empty:
  try:
    parties_list = simulated_polls_final.columns.tolist()
    num_parties = len(parties_list)
    num_sim_runs = len(simulated_polls_final)

    # --- Ranks ---
    print("  Calculating ranks...")
    ranks_df = simulated_polls_final.rank(axis=1, ascending=False, method='first')
    ranks_stats_list = []
    for i in range(1, num_parties + 1):
      prob_rank_i = (ranks_df <= i).sum() / num_sim_runs
      prob_rank_i.name = str(i)
      ranks_stats_list.append(prob_rank_i)
    if ranks_stats_list:
      ranks_stats_final = pd.concat(ranks_stats_list, axis=1)

    # --- Intervals ---
    print("  Calculating interval probabilities...")
    interval_points = np.concatenate((np.arange(0, INTERVAL_MAX + 0.5, 0.5), np.array(ADDITIONAL_POINTS)))
    interval_points = np.unique(interval_points) # Ensure unique and sorted
    interval_points.sort()
    probs_stats_list = []
    for i in interval_points:
      prob_over_i = (simulated_polls_final > (i / 100.0)).sum() / num_sim_runs
      prob_over_i.name = i # Use interval value as name/index later
      probs_stats_list.append(prob_over_i)
    if probs_stats_list:
      probs_stats_final = pd.concat(probs_stats_list, axis=1).T # Transpose, index is interval
      probs_stats_final.index.name = 'Pr'
      probs_interval_final = pd.DataFrame(interval_points, columns=['Pr'])


    # --- Duels ---
    print("  Calculating duels...")
    duels_stats_final = pd.DataFrame(index=parties_list, columns=parties_list, dtype=float)
    for p1 in parties_list:
      for p2 in parties_list:
        # Prob(p1 >= p2)
        duels_stats_final.loc[p1, p2] = (simulated_polls_final[p1] >= simulated_polls_final[p2]).mean()

    # --- Top 2 ---
    print("  Calculating Top 2...")
    top2_condition = (ranks_df <= 2) # True if party is rank 1 or 2
    top2_stats_final = pd.DataFrame(index=parties_list, columns=parties_list, dtype=float)
    for p1 in parties_list:
      for p2 in parties_list:
        if p1 == p2:
          top2_stats_final.loc[p1, p2] = np.nan # Or empty string?
        else:
          # Prob(p1 in Top2 AND p2 in Top2)
          top2_stats_final.loc[p1, p2] = (top2_condition[p1] & top2_condition[p2]).mean()

    # --- Number In ---
    print("  Calculating number of parties in parliament...")
    needs_map = preference_df.set_index('party')['needed']
    # Ensure needs_map covers all parties in simulation, fallback if needed
    needs_aligned = needs_map.reindex(parties_list).fillna(0.05) # Fallback to 5%
    passes_threshold = simulated_polls_final.ge(needs_aligned, axis=1)
    num_in_sim_series = passes_threshold.sum(axis=1)
    value_counts_num_in = num_in_sim_series.value_counts().sort_index()
    # Cumulative probability: P(N >= k)
    cum_prob = value_counts_num_in[::-1].cumsum()[::-1] / num_sim_runs
    # Create final DataFrame
    max_parties_in = num_in_sim_series.max()
    num_in_index = range(0, max_parties_in + 1)
    num_in_stats_final = cum_prob.reindex(num_in_index, fill_value=0.0).reset_index()
    num_in_stats_final.columns = ['index', 'p'] # Match original script output names

    print("Manual statistics calculation complete.")

  except Exception as e:
    print(f"\n>>> Error calculating manual statistics: {e}")
    # Reset outputs
    ranks_stats_final = pd.DataFrame()
    probs_stats_final = pd.DataFrame()
    probs_interval_final = pd.DataFrame()
    duels_stats_final = pd.DataFrame()
    top2_stats_final = pd.DataFrame()
    num_in_stats_final = pd.DataFrame()
else:
  print("Skipping manual statistics: Final simulated polls not available.")


# %%
# --- Step 9: Write Results to Google Sheet ---
print("\n--- 9. Writing Results to Google Sheet ---")
try:
  gc = gspread.service_account()
  sh = gc.open_by_key(SHEET_KEY)
  print(f"Opened GSheet: {sh.title}")
  
  # Helper to write dataframe with custom A1 header
  def write_gsheet_with_a1(worksheet_name, df, a1_header_text, data_start_cell='A2'):
    if df.empty:
      print(f"  Skipping write to '{worksheet_name}': DataFrame is empty.")
      return
    try:
      print(f"  Writing to worksheet: '{worksheet_name}'...")
      wsw = sh.worksheet(worksheet_name)
      wsw.clear()
      # Write custom header to A1
      wsw.update(range_name='A1', values=[[a1_header_text]])
      print(f"    Wrote header '{a1_header_text}' to A1.")
      # Prepare data including header for update
      data_to_write = [df.columns.values.tolist()] + df.values.tolist()
      wsw.update(range_name=data_start_cell, values=data_to_write)
      print(f"    Successfully wrote {len(data_to_write)} rows starting {data_start_cell}.")
    except gspread.exceptions.WorksheetNotFound:
      print(f"  Error: Worksheet '{worksheet_name}' not found.")
    except Exception as e_write:
      print(f"  Error writing to '{worksheet_name}': {e_write}")

  # Helper to write dataframe
  def write_gsheet(worksheet_name, df, start_cell='B1', include_header=True, include_index=False, clear_before_write=True):
    try:
      print(f"  Writing to worksheet: '{worksheet_name}', cell: {start_cell}...")
      wsw = sh.worksheet(worksheet_name)
      if clear_before_write:
        wsw.clear() # Clear sheet before writing
      # Prepare data list including header/index if needed
      data_to_write = []
      if include_header: data_to_write.append(df.columns.tolist())
      if include_index:
        # Need to combine index and values carefully
        df_reset = df.reset_index()
        if include_header: data_to_write = [df_reset.columns.tolist()] + df_reset.values.tolist()
        else: data_to_write = df_reset.values.tolist()
      else:
        data_to_write.extend(df.values.tolist())

      # Write data
      wsw.update(range_name=start_cell, values=data_to_write)
      print(f"  Successfully wrote {len(data_to_write)} rows to '{worksheet_name}'.")
    except gspread.exceptions.WorksheetNotFound:
      print(f"  Error: Worksheet '{worksheet_name}' not found.")
    except Exception as e_write:
      print(f"  Error writing to '{worksheet_name}': {e_write}")

  # --- Write Specific Stats ---
  # Ranks (pořadí_aktuální_aging_cov) - Index is party, Columns are ranks '1', '2', ...
  if not ranks_stats_final.empty:
    write_gsheet(RANKS_AGING_COV_WS, ranks_stats_final.T, start_cell='B1', include_header=True, include_index=False, clear_before_write=False) # Transposed in original

  # Probabilities (pravděpodobnosti_aktuální_aging_cov) - Index is Pr, Columns are parties
  if not probs_stats_final.empty and not probs_interval_final.empty:
    # Write interval points to column A first
    try:
      wsw_prob = sh.worksheet(PROBS_AGING_COV_WS)
      wsw_prob.clear()
      wsw_prob.update(range_name='A2', values=probs_interval_final.values.tolist())
      print(f"  Wrote interval points to '{PROBS_AGING_COV_WS}' column A.")
      # Then write the probabilities starting from B1 (header + data)
      write_gsheet(PROBS_AGING_COV_WS, probs_stats_final, start_cell='B1', include_header=True, include_index=False, clear_before_write=False)
    except Exception as e_prob: print(f" Error writing probabilities: {e_prob}")


  # Duels (duely_aging_cov) - Index is party, Columns are party
  if not duels_stats_final.empty:
    # Write index (row headers) to A3:
    try:
      wsw_duel = sh.worksheet(DUELS_AGING_COV_WS)
      wsw_duel.clear()
      row_headers = [[h] for h in duels_stats_final.index.tolist()] # List of lists
      wsw_duel.update(range_name='A3', values=row_headers)
      print(f"  Wrote row headers to '{DUELS_AGING_COV_WS}' column A.")
      # Write data + column headers starting B2
      write_gsheet(DUELS_AGING_COV_WS, duels_stats_final, start_cell='B2', include_header=True, include_index=False)
      # Write to A1 "Pr[row >= column]"
      wsw_duel.update(range_name='A1', values=[['Pr[row >= column]']])
      # Write starting A3 down the name of the parties in one write
      party_names = duels_stats_final.index.tolist()
      parties_to_write = [[party_names[i]] for i in range(len(party_names))]
      wsw_duel.update(range_name='A3', values=parties_to_write)
    except Exception as e_duel: print(f" Error writing duels: {e_duel}")

  # Top 2 (top_2_cov) - Index is party, Columns are party
  if not top2_stats_final.empty:
    # Write index (row headers) to A3:
    try:
      wsw_top2 = sh.worksheet(TOP2_COV_WS)
      wsw_top2.clear()
      row_headers_top2 = [[h] for h in top2_stats_final.index.tolist()]
      wsw_top2.update(range_name='A3', values=row_headers_top2)
      print(f"  Wrote row headers to '{TOP2_COV_WS}' column A.")
      # Write data + column headers starting B2
      top2_write = top2_stats_final.fillna('') # Replace NaN with empty string for GSheet
      write_gsheet(TOP2_COV_WS, top2_write, start_cell='B2', include_header=True, include_index=False)
      # Write to A1 "Pr[TOP 2]"
      wsw_top2.update(range_name='A1', values=[['Pr[TOP 2]']])
      # Write starting A3 down the name of the parties in one write
      party_names = top2_stats_final.index.tolist()
      parties_to_write = [[party_names[i]] for i in range(len(party_names))]
      wsw_top2.update(range_name='A3', values=parties_to_write)
    except Exception as e_top2: print(f" Error writing Top 2: {e_top2}")


  # Number In (number_in_aging_cov) - Columns are 'index', 'p'
  if not num_in_stats_final.empty:
    # Write header + data starting A1 (assuming sheet clear)
    write_gsheet(NUM_IN_AGING_COV_WS, num_in_stats_final, start_cell='A1', include_header=True, include_index=False)
    
  # --- Write NEW Coalition Probabilities ---
    if not coalitions_exclusive.empty:
      write_gsheet_with_a1(COALITIONS_EXCL_WS, coalitions_exclusive,
                            "Pr [tato koalice nebo libovolná podmnožina]", data_start_cell='A2')
    if not coalitions_inclusive.empty:
      write_gsheet_with_a1(COALITIONS_INCL_WS, coalitions_inclusive,
                            "Pr [přesně tato koalice]", data_start_cell='A2')

    # --- Write NEW Probability In ---
    if not stats_df.empty and 'party' in stats_df.columns and 'in' in stats_df.columns:
      prob_in_df = stats_df[['party', 'in']].sort_values('in', ascending=False).reset_index(drop=True)
      write_gsheet_with_a1(PROB_IN_WS, prob_in_df, "Pr[in]", data_start_cell='A2')
    elif stats_df.empty: print(f"  Skipping write to '{PROB_IN_WS}': stats_df is empty.")
    else: print(f"  Skipping write to '{PROB_IN_WS}': 'party' or 'in' column missing in stats_df.")

  # --- Update Timestamp ---
  try:
    wsw_pref = sh.worksheet(PREF_WRITEBACK_WS)
    timestamp = datetime.datetime.now().isoformat()
    wsw_pref.update(range_name='E2', values=[[timestamp]]) # Cell for timestamp
    print(f"Updated timestamp in '{PREF_WRITEBACK_WS}'.")
  except Exception as e_ts: print(f" Error updating timestamp: {e_ts}")

  # --- Write History Row ---
  if not preference_df.empty and sheet_date:
    try:
      wsh_hist = sh.worksheet(HISTORY_WRITEBACK_WS)
      timestamp = datetime.datetime.now().isoformat()
      # Prepare row data: timestamp, sheet_date, gains..., '', volatilities...
      gains = preference_df.set_index('party').loc[mu_latest_aligned.index]['p'] * 100 # Use aligned order
      volas = preference_df.set_index('party').loc[mu_latest_aligned.index]['volatilita']
      hist_row_data = [timestamp, sheet_date.isoformat()] + gains.tolist() + [''] + volas.tolist()
      wsh_hist.insert_row(hist_row_data, 2, value_input_option='USER_ENTERED') # Insert as second row
      print(f"Inserted history row into '{HISTORY_WRITEBACK_WS}'.")
    except Exception as e_hist: print(f" Error writing history row: {e_hist}")


  print("Finished writing results to Google Sheet.")

except Exception as e_main_write:
  print(f"\n>>> Error interacting with Google Sheet for writing: {e_main_write}")


# %%
print("\n--- Manual Analysis Run Finished ---")