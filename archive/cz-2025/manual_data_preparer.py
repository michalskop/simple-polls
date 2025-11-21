# manual_data_preparer.py
# Functions specific to preparing simulation inputs from manual Google Sheet data.

import pandas as pd
import numpy as np
import gspread
import math
import datetime
from typing import Tuple, List, Optional, Dict, Callable

def load_manual_poll_data(
  sheetkey: str,
  worksheet_name: str = 'preference'
) -> Tuple[pd.DataFrame, datetime.date]:
  """
  Loads manual poll preferences from a Google Sheet.

  Args:
    sheetkey: The key of the Google Spreadsheet.
    worksheet_name: The name of the worksheet containing the preferences.
                    Expected columns: 'party', 'gain' (%), 'date', 'volatilita', 'needed' (%).

  Returns:
    A tuple containing:
    - preference_df: DataFrame with 'party', 'p' (gain 0-1), 'volatilita', 'needed' (0-1), 'date'.
    - today_date: The date extracted from the sheet (assumed consistent).

  Raises:
    ValueError: If required columns are missing.
    Exception: Propagates gspread errors.
  """
  print(f"Loading manual preferences from GSheet key: {sheetkey}, worksheet: {worksheet_name}")
  try:
    gc = gspread.service_account()
    sh = gc.open_by_key(sheetkey)
    ws = sh.worksheet(worksheet_name)
    df = pd.DataFrame(ws.get_all_records())
  except gspread.exceptions.APIError as e:
    print(f"GSpread API Error: {e}")
    raise
  except Exception as e:
    print(f"Error loading GSheet: {e}")
    raise

  required_cols = ['party', 'gain', 'date', 'volatilita', 'needed']
  missing_cols = [col for col in required_cols if col not in df.columns]
  if missing_cols:
    raise ValueError(f"Missing required columns in '{worksheet_name}' sheet: {missing_cols}")

  # Convert percentages, handle potential errors
  df['p'] = pd.to_numeric(df['gain'], errors='coerce') / 100.0
  df['needed'] = pd.to_numeric(df['needed'], errors='coerce') / 100.0
  df['volatilita'] = pd.to_numeric(df['volatilita'], errors='coerce')

  # Extract and validate date
  try:
    # Assuming date is in the first row and consistent
    today_date_str = df['date'].iloc[0]
    today_date = datetime.date.fromisoformat(today_date_str)
    print(f"Extracted date from sheet: {today_date}")
  except (IndexError, ValueError, TypeError) as e:
    print(f"Warning: Could not parse date from sheet ({e}). Using current system date.")
    today_date = datetime.date.today()

  # Keep only relevant columns
  preference_df = df[['party', 'p', 'volatilita', 'needed', 'date']].copy()

  # Check for NaNs introduced by coercion
  if preference_df[['p', 'needed', 'volatilita']].isna().any().any():
    print("Warning: Found NaN values after converting gain/needed/volatilita. Check sheet data.")
    print(preference_df[preference_df[['p', 'needed', 'volatilita']].isna().any(axis=1)])

  print("Manual preference data loaded.")
  return preference_df, today_date


def calculate_aging_coeff(
  date1: datetime.date,
  date2: datetime.date,
  power: float = 1.15
) -> float:
  """Calculates an aging coefficient based on time difference."""
  diff = abs((date2 - date1).days)
  if diff <= 0:
      return 1.0
  # Original formula: pow(diff, 1.15) / diff = diff^0.15
  # Let's use the power directly for clarity: diff^power
  # Correction: original formula was diff^1.15 / diff = diff^0.15
  # Let's re-implement the original intent: pow(diff, power) / diff = diff**(power - 1)
  if power == 1: return 1.0 # Avoid 0^0 issues if power=1
  return math.pow(diff, power - 1) if diff > 0 else 1.0


def calculate_manual_sigman(
  preference_df: pd.DataFrame,
  sample_n: int,
  sheet_date: Optional[datetime.date] = None,
  election_date: Optional[str] = None,
  aging_power: float = 1.15
) -> Tuple[pd.Series, Optional[pd.Series]]:
  """
  Calculates sigma * sqrt(n) based on manual preferences and volatility.
  Optionally calculates an aged version.

  Args:
    preference_df: DataFrame from load_manual_poll_data ('party', 'p', 'volatilita').
    sample_n: The base sample size assumption for binomial error.
    sheet_date (Optional): The date associated with the preference data.
    election_date (Optional): ISO format string of the election date for aging.
    aging_power (float): The exponent used in the aging calculation (e.g., 1.15).

  Returns:
    A tuple containing:
    - sigman_latest: Series of sigma*sqrt(n), indexed by party.
    - sigman_latest_aged (Optional): Series of aged sigma*sqrt(n), or None if no aging.

  Raises:
    ValueError: If required columns are missing or sample_n is invalid.
  """
  required_cols = ['party', 'p', 'volatilita']
  missing_cols = [col for col in required_cols if col not in preference_df.columns]
  if missing_cols:
    raise ValueError(f"Missing required columns in preference_df: {missing_cols}")
  if sample_n <= 0:
    raise ValueError("sample_n must be positive.")

  print("Calculating sigman based on manual preferences...")
  df = preference_df.copy()

  # Calculate standard deviation (sigma) based on binomial approx and volatility
  # sigma = sqrt(p * (1-p) / n) * volatility
  # sigman = sigma * sqrt(n) = sqrt(p * (1-p)) * volatility
  df['p_clipped'] = df['p'].clip(0.001, 0.999) # Clip to avoid sqrt(0) issues near 0/1
  df['sigman_base'] = np.sqrt(df['p_clipped'] * (1 - df['p_clipped']))
  df['sigman'] = df['sigman_base'] * df['volatilita']

  sigman_latest = df.set_index('party')['sigman']
  sigman_latest_aged = None

  # Calculate aged version if dates provided
  if sheet_date and election_date:
    try:
      election_day_dt = datetime.date.fromisoformat(election_date)
      aging_factor = calculate_aging_coeff(sheet_date, election_day_dt, power=aging_power)
      print(f"Applying aging factor: {aging_factor:.4f} (Power: {aging_power})")
      df['sigman_aged'] = df['sigman'] * aging_factor
      sigman_latest_aged = df.set_index('party')['sigman_aged']
    except (ValueError, TypeError) as e:
      print(f"Warning: Could not calculate aging factor ({e}). Skipping aged sigman.")

  print("Sigman calculation complete.")
  return sigman_latest, sigman_latest_aged


def load_manual_correlation(
  sheetkey: str,
  party_order: List[str],
  worksheet_name: str = 'median correlations'
) -> pd.DataFrame:
  """
  Loads a manual correlation matrix from a Google Sheet.

  Args:
    sheetkey: The key of the Google Spreadsheet.
    party_order: The desired order of parties for rows/columns, matching sigman/mu.
    worksheet_name: The name of the worksheet containing the correlation matrix.
                    First column should be party names (index), subsequent columns
                    should be party names (headers).

  Returns:
    corr_matrix: DataFrame representing the correlation matrix, reindexed and
                 reordered according to party_order, NaNs filled with 0.

  Raises:
    ValueError: If the matrix structure is incorrect or parties mismatch.
    Exception: Propagates gspread errors.
  """
  print(f"Loading manual correlation from GSheet key: {sheetkey}, worksheet: {worksheet_name}")
  try:
    gc = gspread.service_account()
    sh = gc.open_by_key(sheetkey)
    ws = sh.worksheet(worksheet_name)
    df = pd.DataFrame(ws.get_all_records())
  except gspread.exceptions.APIError as e:
      print(f"GSpread API Error: {e}")
      raise
  except Exception as e:
      print(f"Error loading GSheet: {e}")
      raise

  if df.empty:
    raise ValueError("Correlation sheet is empty.")

  # Assume first column is the index (party names)
  index_col_name = df.columns[0]
  df = df.set_index(index_col_name)

  # Convert to numeric, coercing errors
  df = df.apply(pd.to_numeric, errors='coerce')

  # Check if columns match index
  if not df.columns.equals(df.index):
    print("Warning: Correlation matrix columns do not exactly match index.")
    print(f"Index: {df.index.tolist()}")
    print(f"Columns: {df.columns.tolist()}")

  # Reindex and reorder based on the provided party order
  try:
    corr_matrix = df.reindex(index=party_order, columns=party_order)
  except KeyError as e:
    raise ValueError(f"Party '{e}' from party_order not found in correlation matrix index/columns.")

  # Fill NaNs (e.g., if matrix was incomplete or parties missing) with 0
  # Ensure diagonal is 1
  corr_matrix = corr_matrix.fillna(0)
  np.fill_diagonal(corr_matrix.values, 1.0)

  print("Manual correlation matrix loaded and processed.")
  return corr_matrix


def add_uniform_noise(
  simulated_polls_mvn: pd.DataFrame,
  preference_df: pd.DataFrame,
  sample_n: int,
  aging_factor: float = 1.0,
  # Original script used coefficients like 1.5 * 0.9 * 0.9 = 1.215
  # Let's make this explicit
  uniform_error_coef: float = 1.215
) -> pd.DataFrame:
  """
  Adds uniform noise to simulated poll results based on preference data.
  This step is specific to the manual CZ-2025 analysis.

  Args:
    simulated_polls_mvn: DataFrame of simulations from multivariate normal.
    preference_df: DataFrame containing 'party', 'p', 'volatilita'.
    sample_n: Base sample size used for sdx calculation.
    aging_factor (float): Aging factor to apply to the noise range. Defaults to 1.0.
    uniform_error_coef (float): Coefficient applied to sdx for uniform noise range.

  Returns:
      DataFrame with uniform noise added.
  """
  if simulated_polls_mvn.empty:
    return simulated_polls_mvn

  print(f"Adding uniform noise (Coef: {uniform_error_coef}, Aging: {aging_factor:.4f})...")
  noisy_simulations = simulated_polls_mvn.copy()
  pref_map = preference_df.set_index('party')

  for party in noisy_simulations.columns:
    if party in pref_map.index:
      p = pref_map.loc[party, 'p']
      volatility = pref_map.loc[party, 'volatilita']
      p_clipped = np.clip(p, 0.001, 0.999)

      # Calculate sdx for this party (sigma scaled by sqrt(n) and volatility)
      # sigma = sqrt(p*(1-p)/n) * volatility
      sdx = math.sqrt(p_clipped * (1 - p_clipped) / sample_n) * volatility

      # Calculate range for uniform distribution (scaled by coef and aging)
      # Range: [-sdx * sqrt(3), sdx * sqrt(3)] * uniform_error_coef * aging_factor
      # Scale factor for half-width: sqrt(3) * uniform_error_coef * aging_factor
      half_width = sdx * math.sqrt(3) * uniform_error_coef * aging_factor
      low = -half_width
      high = half_width

      # Generate and add noise
      noise = np.random.uniform(low=low, high=high, size=len(noisy_simulations))
      noisy_simulations[party] += noise
    else:
      print(f"Warning: Party '{party}' not found in preference data for uniform noise calculation.")

  # Clip again to ensure results are valid percentages
  noisy_simulations = noisy_simulations.clip(0, 1)
  print("Uniform noise added.")
  return noisy_simulations


def load_custom_coalitions_from_sheet(
  sheetkey: str,
  worksheet_name: str = 'vlastní_koalice'
) -> Optional[List[str]]:
  """
  Loads custom coalition definitions from a Google Sheet.
  Assumes each row represents a coalition, with party names in columns starting from the first.

  Args:
    sheetkey: The key of the Google Spreadsheet.
    worksheet_name: The name of the worksheet containing the coalitions.

  Returns:
    A list of coalition strings (e.g., ['PartyA*PartyB', 'PartyC*PartyD*PartyE']),
    or None if loading fails or the sheet is empty.
  """
  print(f"Loading custom coalitions from GSheet key: {sheetkey}, worksheet: {worksheet_name}")
  try:
    gc = gspread.service_account()
    sh = gc.open_by_key(sheetkey)
    ws = sh.worksheet(worksheet_name)
    # Get all values, including empty strings
    all_values = ws.get_all_values()
  except gspread.exceptions.APIError as e:
    print(f"GSpread API Error: {e}")
    return None
  except gspread.exceptions.WorksheetNotFound:
    print(f"Warning: Custom coalitions worksheet '{worksheet_name}' not found. Skipping.")
    return None
  except Exception as e:
    print(f"Error loading custom coalitions GSheet: {e}")
    return None

  if not all_values or len(all_values) <= 1: # Check for header + data
    print(f"Warning: Custom coalitions sheet '{worksheet_name}' appears empty or has only headers. Skipping.")
    return None

  custom_coalitions_list = []
  # Start from second row (index 1) assuming first row is header
  for row in all_values[1:]:
    # Filter out empty strings and join non-empty party names with '*'
    parties = [cell.strip() for cell in row if cell.strip()]
    if parties: # Only add if there are party names in the row
      coalition_str = '*'.join(parties)
      custom_coalitions_list.append(coalition_str)

  print(f"Loaded {len(custom_coalitions_list)} custom coalitions.")
  print(f"  Examples: {custom_coalitions_list[:5]}")
  return custom_coalitions_list


def load_partner_definitions(
  sheetkey: str,
  worksheet_name: str = 'partners'
) -> Optional[Dict[str, List[str]]]:
  """
  Loads main party and potential partner definitions from a Google Sheet.
  Assumes the first column is the main party and subsequent columns on the same row
  are its potential partners.

  Args:
    sheetkey: The key of the Google Spreadsheet.
    worksheet_name: The name of the worksheet containing the partner definitions.

  Returns:
    A dictionary where keys are main party names and values are lists of
    their potential partner names, or None if loading fails or sheet is empty.
  """
  print(f"Loading partner definitions from GSheet key: {sheetkey}, worksheet: {worksheet_name}")
  try:
    gc = gspread.service_account()
    sh = gc.open_by_key(sheetkey)
    ws = sh.worksheet(worksheet_name)
    # Get all values, including empty strings
    all_values = ws.get_all_values()
  except gspread.exceptions.APIError as e:
      print(f"GSpread API Error: {e}")
      return None
  except gspread.exceptions.WorksheetNotFound:
      print(f"Warning: Partner definitions worksheet '{worksheet_name}' not found. Skipping.")
      return None
  except Exception as e:
      print(f"Error loading partner definitions GSheet: {e}")
      return None

  if not all_values or len(all_values) <= 1: # Check for header + data
    print(f"Warning: Partner definitions sheet '{worksheet_name}' appears empty or has only headers. Skipping.")
    return None

  partner_definitions = {}
  # Start from second row (index 1) assuming first row is header
  for row in all_values[1:]:
    if not row or not row[0].strip(): # Skip empty rows or rows without a main party
        continue
    main_party = row[0].strip()
    # Get partners from subsequent columns, filtering empty strings
    partners = [cell.strip() for cell in row[1:] if cell.strip()]
    partner_definitions[main_party] = partners

  if not partner_definitions:
      print(f"Warning: No valid partner definitions found in '{worksheet_name}'.")
      return None

  print(f"Loaded partner definitions for {len(partner_definitions)} main parties.")
  # print(f"  Definitions: {partner_definitions}") # Optional: print dictionary
  return partner_definitions