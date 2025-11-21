"""Simulations for NL-2025 (version 2)."""

import gspread
import pandas as pd
import datetime
import math
import numpy as np
import scipy.stats

SHEET_KEY = '1VbbRpS7lDBY-6fl7GsoHOq8lZjmn8L2nUgO9eEwlyW8'
TAB_NAME = 'preference'
CORR_TAB_NAME = 'correlations'
INTERVAL_TAB_NAME = 'intervals'
RANKS_TAB_NAME = 'ranks'
RANKS_NO_COV_TAB_NAME = 'ranks-no-cov'
TOP5_TAB_NAME = 'top5'
DUELS_TAB_NAME = 'duels'
ELECTION_DATE = '2025-10-29'
SAMPLE_N = 1000      # Used in statistical error calculation
SAMPLE_SIM = 2000    # Number of simulations
NUM_SEATS = 150      # Total number of seats in the parliament

# --- Step 1: Load Data ---
def load_data(sheetkey, tab_name):
    """Loads data from a Google Sheet into a pandas DataFrame."""
    print(f"Loading data from sheet '{sheetkey}' and tab '{tab_name}'...")
    try:
        gc = gspread.service_account()
        sh = gc.open_by_key(sheetkey)
        ws = sh.worksheet(tab_name)
        df = pd.DataFrame(ws.get_all_records())
        print("Data loaded successfully.")
        return df
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"Error: Spreadsheet with key '{sheetkey}' not found.")
        return None
    except gspread.exceptions.WorksheetNotFound:
        print(f"Error: Worksheet with name '{tab_name}' not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- Step 2: Process Data ---
def process_data(df):
    """Processes the raw preference data and checks sums."""
    print("\n--- Step 2: Processing Data ---")
    
    # Extract total valid votes from the first row of its column
    try:
        # Ensure the column is treated as a string, remove commas, then convert
        total_valid_votes = int(str(df['total_valid_votes'].iloc[0]).replace(',', ''))
        print(f"Total valid votes from sheet: {total_valid_votes}")
    except (KeyError, ValueError, IndexError):
        print("Warning: 'total_valid_votes' column not found or invalid. Using 0.")
        total_valid_votes = 0

    df['p'] = df['gain'] / 100
    poll_date = datetime.date.fromisoformat(df['date'][0])
    print(f"Poll date set to: {poll_date}")
    
    parties_p_sum = df['p'].sum()
    print(f"Sum of 'p' for listed parties: {parties_p_sum:.4f}")

    total_p_sum = parties_p_sum
    others_p = 0.0
    if 'others' in df.columns and df['others'].iloc[0] != '':
        try:
            others_val = str(df['others'].iloc[0]).replace(',', '.')
            others_p = float(others_val) / 100
            print(f"Value from 'others' column: {others_p:.4f}")
            total_p_sum += others_p
        except (ValueError, IndexError):
            print("Could not process 'others' column value.")

    print(f"--> Total sum including Others: {total_p_sum:.4f}")
    if not math.isclose(total_p_sum, 1.0):
        print(f"Warning: Total sum is not 1.0 (Difference: {1.0 - total_p_sum:.4f})")
        
    return df, poll_date, total_valid_votes, others_p

# --- Step 3: Define Aging Coefficient ---
def aging_coeff(day1, day2):
    """Calculates the aging coefficient based on the difference between two dates."""
    diff = abs((day2 - day1).days)
    if diff <= 0:
        return 1
    return pow(diff, 1.15) / diff

# --- Step 4: Define Error Functions ---
def normal_error(p, n, volatility, coef=1):
    """Calculates normal (statistical) error."""
    p['sdx'] = (n * p['p'] * (1 - p['p'])).apply(math.sqrt) / n * coef * volatility
    p['normal_error'] = scipy.stats.norm.rvs(loc=0, scale=p['sdx'])
    return p

def uniform_error(p, n, volatility, coef=1):
    """Calculates uniform error. Relies on 'sdx' from normal_error."""
    if 'sdx' not in p.columns:
        raise ValueError("Column 'sdx' not found. Run normal_error first.")
    p['uniform_error'] = scipy.stats.uniform.rvs(loc=(-1 * p['sdx'] * math.sqrt(3) * coef), scale=(2 * p['sdx'] * math.sqrt(3) * coef))
    return p

# --- Step 5: Run Core Simulation ---
def run_simulation(df, num_simulations, sample_n, aging_coeff):
    """Runs the core simulation loop without correlations."""
    print(f"\n--- Step 5: Running {num_simulations} Standard Simulations ---")
    results = []
    results_aging = []

    for i in range(num_simulations):
        p_iter = df.copy()
        p_iter = normal_error(p_iter, sample_n, p_iter['volatilita'], 0.9)
        p_iter = uniform_error(p_iter, sample_n, p_iter['volatilita'], 1.5 * 0.9 * 0.9)
        p_iter['estimate'] = p_iter['normal_error'] + p_iter['uniform_error'] + p_iter['p']
        p_iter['estimate_aging'] = aging_coeff * (p_iter['normal_error'] + p_iter['uniform_error']) + p_iter['p']
        
        results.append(dict(zip(p_iter['party'], p_iter['estimate'])))
        results_aging.append(dict(zip(p_iter['party'], p_iter['estimate_aging'])))

    simulations = pd.DataFrame(results)
    simulations_aging = pd.DataFrame(results_aging)

    # Clip results at 0 to prevent negative vote shares
    simulations = simulations.clip(lower=0)
    simulations_aging = simulations_aging.clip(lower=0)

    print("Standard simulations completed.")
    return simulations, simulations_aging

# --- Step 6: Load Correlation Data ---
def load_correlation_data(sheetkey, tab_name):
    """Loads the correlation matrix."""
    print(f"\n--- Step 6: Loading Correlation Data from tab '{tab_name}' ---")
    corr_df = load_data(sheetkey, tab_name)
    if corr_df is not None:
        # Use the first column as the index, whatever its name is.
        index_col_name = corr_df.columns[0]
        corr_df = corr_df.set_index(index_col_name)
        # Explicitly convert all correlation data to numeric, coercing errors.
        corr_df = corr_df.apply(pd.to_numeric, errors='coerce')
        corr_df.columns = corr_df.index
        # Force the entire dataframe to a consistent float type to avoid object dtype issues.
        corr_df = corr_df.astype('float64')
        print(f"Correlation matrix loaded, converted to numeric, and indexed using column '{index_col_name}'.")
        return corr_df
    return None

# --- Step 7: Run Simulation with Covariance ---
def run_simulation_with_covariance(df, corr, num_simulations, aging_coeff):
    """Runs the simulation using a covariance matrix. REWRITTEN FOR DEBUGGING."""
    print(f"\n--- Step 7: Running {num_simulations} Simulations with Covariance (DEBUG REWRITE) ---")
    
    parties = df['party'].tolist()
    corr = corr.loc[parties, parties]
    
    # 1. Calculate SDX and Random Error
    df['sdx'] = (SAMPLE_N * df['p'] * (1 - df['p'])).apply(math.sqrt) / SAMPLE_N * df['volatilita'] * 0.9
    df['random_error_val'] = df['sdx'] * 1.5 * 0.9 * 0.9

    # 2. Build Covariance Matrix
    n_parties = len(parties)
    sdx_map = df.set_index('party')['sdx']
    cov_matrix = np.zeros((n_parties, n_parties))
    for i in range(n_parties):
        for j in range(n_parties):
            party_i = parties[i]
            party_j = parties[j]
            cov_matrix[i, j] = corr.loc[party_i, party_j] * sdx_map[party_i] * sdx_map[party_j]

    # 3. Correct Matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    if np.any(eigenvalues < 0):
        print("Warning: Covariance matrix not positive semi-definite. Correcting.")
        eigenvalues[eigenvalues < 0] = 0
        cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # 4. Run Simulation Loop
    results_cov = []
    results_aging_cov = []
    base_p = df.set_index('party')['p']
    random_error_vals = df.set_index('party')['random_error_val']

    for _ in range(num_simulations):
        # Generate errors for this iteration
        mvn_errors = np.random.multivariate_normal(np.zeros(n_parties), cov_matrix)
        uniform_errors = np.random.uniform(-1, 1, n_parties) * random_error_vals.values

        # Combine errors and calculate estimates
        total_error = mvn_errors + uniform_errors
        
        estimate = base_p.values + total_error
        estimate_aging = base_p.values + (aging_coeff * total_error)
        
        results_cov.append(dict(zip(parties, estimate)))
        results_aging_cov.append(dict(zip(parties, estimate_aging)))

    simulations_cov = pd.DataFrame(results_cov)
    simulations_aging_cov = pd.DataFrame(results_aging_cov)

    # Clip results at 0 to prevent negative vote shares
    simulations_cov = simulations_cov.clip(lower=0)
    simulations_aging_cov = simulations_aging_cov.clip(lower=0)

    print("Simulations with covariance completed.")
    return simulations_cov, simulations_aging_cov

# --- Step 8: Calculate First Scrutinium (Hare Method) ---
def calculate_hare_seats(sims_df, num_seats):
    """Calculates the number of seats won in the first round using the Hare quota."""
    print(f"\n--- Step 8: Calculating First Scrutinium Seats (Hare Method) ---")
    # The simulation results are proportions (e.g., 0.25 for 25%), so the total is 1.0
    hare_quota = 1.0 / num_seats
    print(f"Hare Quota (for proportions) calculated: {hare_quota:.6f}")
    
    # Divide simulation results by the quota and take the integer part
    seats_df = (sims_df / hare_quota).apply(np.floor).astype(int)
    
    print("First scrutinium seats calculated.")
    return seats_df

# --- Step 9: Calculate Second Scrutinium (D'Hondt Method) ---
def calculate_d_hondt_seats(sims_df, initial_seats_df, total_votes, num_seats):
    """Allocates remaining seats using the D'Hondt (largest averages) method."""
    print("\n--- Step 9: Calculating Second Scrutinium Seats (D'Hondt Method) ---")
    
    # Convert proportions to absolute votes for D'Hondt calculation. 
    # The sims_df is now correctly pre-normalized before being passed to this function.
    sim_votes_df = (sims_df * total_votes).round()

    final_seats_list = []
    # Iterate over each simulation (row)
    for i in range(len(sim_votes_df)):
        votes = sim_votes_df.iloc[i]
        current_seats = initial_seats_df.iloc[i].copy()
        
        seats_allocated = current_seats.sum()
        remaining_seats = num_seats - seats_allocated

        # Parties must have >= 1 initial seat to compete for remaining seats
        eligible_parties = current_seats[current_seats >= 1].index

        if remaining_seats > 0 and not eligible_parties.empty:
            for _ in range(remaining_seats):
                # Calculate averages for eligible parties only
                averages = votes[eligible_parties] / (current_seats[eligible_parties] + 1)
                # Find the party with the highest average
                winner = averages.idxmax()
                # Award the seat
                current_seats[winner] += 1
        
        final_seats_list.append(current_seats)

    final_seats_df = pd.DataFrame(final_seats_list, index=initial_seats_df.index)
    print("Second scrutinium seats calculated.")
    return final_seats_df

# --- Step 10: Calculate Rank Probabilities ---
def calculate_and_print_rank_probabilities(seats_df, simulation_type):
    """Calculates and prints the probability for each party to achieve a certain rank."""
    print(f"\n\n--- Step 10: Rank Probabilities for {simulation_type} Simulation ---")

    def get_ranks(row):
        # Create a DataFrame from the row to sort by seats (desc) and party name (asc)
        temp_df = pd.DataFrame({'seats': row.values, 'party': row.index})
        temp_df = temp_df.sort_values(by=['seats', 'party'], ascending=[False, True])
        # Assign rank based on the sorted order
        temp_df['rank'] = range(1, len(temp_df) + 1)
        # Return a Series of ranks indexed by party name
        return temp_df.set_index('party')['rank']

    # Apply the ranking function to every simulation (row)
    ranks_df = seats_df.apply(get_ranks, axis=1)

    # Calculate the probability of each rank for each party
    all_ranks = ranks_df.stack().reset_index()
    all_ranks.columns = ['simulation_index', 'party', 'rank']

    # Group by party and rank, then count the occurrences
    rank_counts = all_ranks.groupby(['party', 'rank']).size().unstack(fill_value=0)

    # Convert counts to probabilities
    rank_probabilities = rank_counts / len(seats_df)

    print("Probability of each party achieving a given rank:")
    # Print with formatting to make it readable
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
        print(rank_probabilities.to_string(float_format="{:.3f}".format))

# --- Step 11: Calculate Interval Probabilities ---
def calculate_and_print_interval_probabilities(seats_df, props_df, simulation_type, output_column_name):
    """Loads intervals and calculates the probability of results falling within them."""
    print(f"\n\n--- Step 11: Interval Probabilities for {simulation_type} Simulation ---")

    # 1. Load interval data
    intervals_df = load_data(SHEET_KEY, INTERVAL_TAB_NAME)
    if intervals_df is None:
        print("Could not load interval data. Skipping calculation.")
        return

    print("Calculating and writing probabilities for the following intervals:")
    
    # Prepare a list to hold the calculated probabilities
    probabilities = [None] * len(intervals_df)

    # 2. Iterate through each interval rule
    for index, row in intervals_df.iterrows():
        party = row['party']
        limit = row['limits']
        limit_type = row['type']
        prob = None

        try:
            lower_bound, upper_bound = map(float, limit.split('-'))
            
            if limit_type == 'seats':
                if party in seats_df.columns:
                    party_results = seats_df[party]
                    prob = ((party_results >= lower_bound) & (party_results <= upper_bound)).mean()

            elif limit_type == 'percent':
                if party in props_df.columns:
                    party_results = props_df[party] * 100
                    prob = ((party_results >= lower_bound) & (party_results <= upper_bound)).mean()
            
            probabilities[index] = prob

        except (ValueError, KeyError):
            # Keep prob as None if there's an error
            pass 

    # Add the new probabilities as a column to the dataframe
    intervals_df[output_column_name] = probabilities
    print(intervals_df[['party', 'limits', 'type', output_column_name]].to_string())

    # 3. Write the updated column back to the sheet
    try:
        gc = gspread.service_account()
        sh = gc.open_by_key(SHEET_KEY)
        ws = sh.worksheet(INTERVAL_TAB_NAME)

        # Find the column index for the output column
        header = ws.row_values(1)
        try:
            col_index = header.index(output_column_name) + 1
        except ValueError:
            # Column doesn't exist, add it to the end
            col_index = len(header) + 1
            ws.update_cell(1, col_index, output_column_name)

        # Prepare the data for updating (needs to be a list of lists)
        update_values = [[p] if p is not None else [''] for p in probabilities]
        
        # Update the column cells, starting from the second row
        ws.update(update_values, range_name=f'{gspread.utils.rowcol_to_a1(2, col_index)}')
        print(f"Successfully wrote probabilities to column '{output_column_name}' in tab '{INTERVAL_TAB_NAME}'.")

    except Exception as e:
        print(f"An error occurred while writing to Google Sheets: {e}")

# --- Step 12: Calculate Top 5 Rank Probabilities ---
def calculate_and_print_top5_rank_probabilities(seats_df, simulation_type):
    """Calculates rank probabilities for a specific subset of 5 parties."""
    print(f"\n\n--- Step 12: Top 5 Rank Probabilities for {simulation_type} Simulation ---")

    top_5_parties = ['VVD', 'D66', 'CDA', 'GL-PvdA', 'PVV']
    
    # Filter the seats dataframe to only include the top 5 parties
    filtered_seats_df = seats_df[top_5_parties]

    def get_top5_ranks(row):
        # Create a DataFrame from the row to sort
        temp_df = pd.DataFrame({'seats': row.values, 'party': row.index})
        # Sort by seats (desc) and then party name (desc - reverse alphabetical)
        temp_df = temp_df.sort_values(by=['seats', 'party'], ascending=[False, False])
        # Assign rank
        temp_df['rank'] = range(1, len(temp_df) + 1)
        return temp_df.set_index('party')['rank']

    # Apply the ranking function to the filtered dataframe
    ranks_df = filtered_seats_df.apply(get_top5_ranks, axis=1)

    # Calculate probabilities
    all_ranks = ranks_df.stack().reset_index()
    all_ranks.columns = ['simulation_index', 'party', 'rank']
    rank_counts = all_ranks.groupby(['party', 'rank']).size().unstack(fill_value=0)
    rank_probabilities = rank_counts / len(seats_df)

    print("Probability of each of the Top 5 parties achieving a given rank (within the group):")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
        print(rank_probabilities.to_string(float_format="{:.3f}".format))

# --- Step 13: Write Ranks to Google Sheet ---
def write_ranks_to_sheet(seats_df, party_order_df, sheet_key, tab_name):
    """Calculates rank probabilities and writes them to a specified Google Sheet tab."""
    print(f"\n\n--- Step 13: Writing Rank Probabilities to Sheet '{sheet_key}', Tab '{tab_name}' ---")

    # 1. Calculate Rank Probabilities (reusing the same logic as before)
    def get_ranks(row):
        temp_df = pd.DataFrame({'seats': row.values, 'party': row.index})
        temp_df = temp_df.sort_values(by=['seats', 'party'], ascending=[False, True])
        temp_df['rank'] = range(1, len(temp_df) + 1)
        return temp_df.set_index('party')['rank']

    ranks_df = seats_df.apply(get_ranks, axis=1)
    all_ranks = ranks_df.stack().reset_index()
    all_ranks.columns = ['simulation_index', 'party', 'rank']
    rank_counts = all_ranks.groupby(['party', 'rank']).size().unstack(fill_value=0)
    rank_probabilities = rank_counts / len(seats_df)

    # 2. Format the DataFrame for the sheet
    # Transpose so parties are columns and ranks are rows
    output_df = rank_probabilities.T
    
    # Get the original party order
    party_order = party_order_df['party'].tolist()
    
    # Reorder the columns to match the original party order
    # Ensure we only use parties that are present in the simulation results
    existing_parties_in_order = [p for p in party_order if p in output_df.columns]
    output_df = output_df[existing_parties_in_order]
    
    # Add a 'rank' column for the index
    output_df.reset_index(inplace=True)
    output_df = output_df.rename(columns={'index': 'rank'})

    try:
        # 3. Write to Google Sheet
        gc = gspread.service_account()
        sh = gc.open_by_key(sheet_key)
        
        try:
            ws = sh.worksheet(tab_name)
        except gspread.exceptions.WorksheetNotFound:
            print(f"Worksheet '{tab_name}' not found. Creating it.")
            ws = sh.add_worksheet(title=tab_name, rows=100, cols=30)

        # Determine the exact range to clear and update
        num_rows, num_cols = output_df.shape
        # Add 1 to rows for the header
        target_range = f"A1:{gspread.utils.rowcol_to_a1(num_rows + 1, num_cols)}"

        print(f"Clearing target range {target_range} in tab '{tab_name}'...")
        ws.batch_clear([target_range])

        print(f"Writing new rank probabilities to tab '{tab_name}'...")
        ws.update(target_range, [output_df.columns.values.tolist()] + output_df.values.tolist())
        print("Successfully wrote ranks to Google Sheet.")

    except Exception as e:
        print(f"An error occurred while writing to Google Sheets: {e}")

# --- Step 14: Write Top 5 Ranks to Google Sheet ---
def write_top5_ranks_to_sheet(seats_cov_df, seats_std_df, party_order_df, sheet_key, tab_name):
    """Calculates and writes the Top 5 rank probabilities for both simulations to a single sheet."""
    print(f"\n\n--- Step 14: Writing Top 5 Rank Probabilities to Sheet '{sheet_key}', Tab '{tab_name}' ---")

    top_5_parties = ['VVD', 'D66', 'CDA', 'GL-PvdA', 'PVV']

    def _calculate_top5_probs(seats_df):
        """Helper to calculate Top 5 rank probabilities."""
        filtered_seats = seats_df[top_5_parties]
        def get_ranks(row):
            temp_df = pd.DataFrame({'seats': row.values, 'party': row.index})
            temp_df = temp_df.sort_values(by=['seats', 'party'], ascending=[False, False])
            temp_df['rank'] = range(1, len(temp_df) + 1)
            return temp_df.set_index('party')['rank']
        
        ranks_df = filtered_seats.apply(get_ranks, axis=1)
        all_ranks = ranks_df.stack().reset_index()
        all_ranks.columns = ['sim', 'party', 'rank']
        rank_counts = all_ranks.groupby(['party', 'rank']).size().unstack(fill_value=0)
        return rank_counts / len(seats_df)

    # 1. Calculate probabilities for both simulations
    cov_probs = _calculate_top5_probs(seats_cov_df).T
    std_probs = _calculate_top5_probs(seats_std_df).T

    # 2. Get original party order and filter for top 5
    party_order = party_order_df['party'].tolist()
    top_5_order = [p for p in party_order if p in top_5_parties]

    # 3. Reorder the columns of each probability table
    cov_probs = cov_probs[top_5_order]
    std_probs = std_probs[top_5_order]

    # 4. Create label and separator columns filled with empty strings
    cov_label = pd.DataFrame('', index=cov_probs.index, columns=['cov'])
    std_label = pd.DataFrame('', index=std_probs.index, columns=['no-cov'])
    separator = pd.DataFrame('', index=cov_probs.index, columns=[''])

    # 5. Concatenate all parts in the correct order
    output_df = pd.concat([cov_label, cov_probs, separator, std_label, std_probs], axis=1)

    # Add a 'rank' column and set it as the first column
    output_df.reset_index(inplace=True)
    output_df = output_df.rename(columns={'index': 'rank'})

    # 3. Write to Google Sheet
    try:
        gc = gspread.service_account()
        sh = gc.open_by_key(sheet_key)
        try:
            ws = sh.worksheet(tab_name)
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title=tab_name, rows=20, cols=30)

        # Clear and write
        target_range = f"A1:{gspread.utils.rowcol_to_a1(output_df.shape[0] + 1, output_df.shape[1])}"
        print(f"Clearing target range {target_range} in tab '{tab_name}' and writing new data...")
        ws.batch_clear([target_range])
        ws.update(target_range, [output_df.columns.values.tolist()] + output_df.values.tolist(), value_input_option='USER_ENTERED')
        print("Successfully wrote Top 5 ranks to Google Sheet.")

    except Exception as e:
        print(f"An error occurred while writing Top 5 ranks to Google Sheets: {e}")

# --- Step 15: Calculate and Write Summary Statistics ---
def calculate_and_write_summary_stats(final_seats_cov_df, hare_seats_cov_df, final_seats_std_df, sheet_key, tab_name):
    """Calculates and writes a full summary of statistics to the preference tab."""
    print(f"\n\n--- Step 15: Writing Summary Statistics to Tab '{tab_name}' ---")

    # 1. Calculate all statistics
    # Covariance model stats
    stats_cov = pd.DataFrame({
        'average - 1st scrutiny': hare_seats_cov_df.mean(),
        'median - 1st scrutiny': hare_seats_cov_df.median(),
        'average - 2nd scrutiny': final_seats_cov_df.mean(),
        'median - 2nd scrutiny': final_seats_cov_df.median(),
        '95% interval lower': final_seats_cov_df.quantile(0.025),
        '95% interval upper': final_seats_cov_df.quantile(0.975),
        'mode': final_seats_cov_df.mode().iloc[0]
    })

    # Standard model stats
    stats_std = pd.DataFrame({
        'median-no-cov': final_seats_std_df.median()
    })

    # Combine all stats into one DataFrame
    summary_df = pd.concat([stats_cov, stats_std], axis=1).round(2)
    print("Calculated Summary Statistics:")
    print(summary_df)

    # 2. Write to Google Sheet
    try:
        gc = gspread.service_account()
        sh = gc.open_by_key(sheet_key)
        ws = sh.worksheet(tab_name)
        header = ws.row_values(1)
        sheet_parties = [row.get('party') for row in ws.get_all_records()]

        # Prepare for batch update
        cell_updates = []
        for col_name in summary_df.columns:
            try:
                col_index = header.index(col_name) + 1
            except ValueError:
                col_index = len(header) + 1
                ws.update_cell(1, col_index, col_name)
                header.append(col_name)
            
            for party_name, value in summary_df[col_name].items():
                if party_name in sheet_parties:
                    row_index = sheet_parties.index(party_name) + 2 # +2 for header and 1-based index
                    cell_updates.append(gspread.Cell(row_index, col_index, str(value)))

        if cell_updates:
            print(f"Updating {len(cell_updates)} cells across {len(summary_df.columns)} columns...")
            ws.update_cells(cell_updates, value_input_option='USER_ENTERED')
            print("Successfully wrote summary statistics to Google Sheet.")

    except Exception as e:
        print(f"An error occurred while writing summary stats to Google Sheets: {e}")


# --- Step 17: Calculate and Write Duels ---
def calculate_and_write_duels(seats_cov_df, seats_std_df, sheet_key, tab_name):
    """Calculates duel probabilities and writes them back to the duels tab."""
    print(f"\n\n--- Step 17: Calculating and Writing Duel Probabilities to Tab '{tab_name}' ---")

    # 1. Load duel definitions
    duels_df = load_data(sheet_key, tab_name)
    if duels_df is None:
        print("Could not load duels data. Skipping calculation.")
        return

    # 2. Calculate probabilities for each duel
    results = []
    for index, row in duels_df.iterrows():
        p1_name = row.get('party 1')
        p2_name = row.get('party 2')

        if not p1_name or not p2_name:
            results.append([None] * 6) # Append placeholders if parties are not defined
            continue

        # Ensure both parties are in the simulation results
        if p1_name not in seats_cov_df.columns or p2_name not in seats_cov_df.columns:
            results.append([None] * 6)
            continue

        # Covariance simulation results
        p1_wins_cov = (seats_cov_df[p1_name] > seats_cov_df[p2_name]).mean()
        p2_wins_cov = (seats_cov_df[p2_name] > seats_cov_df[p1_name]).mean()
        tie_cov = (seats_cov_df[p1_name] == seats_cov_df[p2_name]).mean()

        # Standard simulation results
        p1_wins_std = (seats_std_df[p1_name] > seats_std_df[p2_name]).mean()
        p2_wins_std = (seats_std_df[p2_name] > seats_std_df[p1_name]).mean()
        tie_std = (seats_std_df[p1_name] == seats_std_df[p2_name]).mean()
        
        results.append([p1_wins_cov, p2_wins_cov, tie_cov, p1_wins_std, p2_wins_std, tie_std])

    # 3. Write results back to the sheet
    results_df = pd.DataFrame(results, columns=['p1', 'p2', 'p0', 'p1-no-cov', 'p2-no-cov', 'p0-no-cov'])
    print("Calculated Duel Probabilities:")
    print(results_df)

    try:
        gc = gspread.service_account()
        sh = gc.open_by_key(sheet_key)
        ws = sh.worksheet(tab_name)
        header = ws.row_values(1)

        # Update each column individually
        for col_name in results_df.columns:
            try:
                col_index = header.index(col_name) + 1
            except ValueError:
                col_index = len(header) + 1
                ws.update_cell(1, col_index, col_name)
                header.append(col_name) # Update header for next iteration

            # Prepare column data for update
            update_values = [[val] if pd.notna(val) else [''] for val in results_df[col_name]]
            if update_values:
                ws.update(update_values, range_name=f'{gspread.utils.rowcol_to_a1(2, col_index)}')
        
        print(f"Successfully wrote duel probabilities to tab '{tab_name}'.")

    except Exception as e:
        print(f"An error occurred while writing duel probabilities to Google Sheets: {e}")

def test_seat_allocation():
    """Tests the seat allocation logic with the 2021 election results."""
    print("\n\n--- Running Seat Allocation Test with 2021 Election Data ---")
    test_data = {
        "People's Party for Freedom and Democracy": 2279130,
        "Democrats 66": 1565861,
        "Party for Freedom": 1124482,
        "Christian Democratic Appeal": 990601,
        "Socialist Party": 623371,
        "Labour Party": 597192,
        "GroenLinks": 537308,
        "Forum for Democracy": 523083,
        "Party for the Animals": 399750,
        "Christian Union": 351275,
        "Volt Netherlands": 252480,
        "JA21": 246620,
        "Reformed Political Party": 215249,
        "Denk": 211237,
        "50PLUS": 106702,
        "Farmer–Citizen Movement": 104319,
        "BIJ1": 87238,
        "Code Orange": 40731,
        "NIDA": 33834,
        "Splinter": 30328,
        "Pirate Party": 22816,
        "JONG": 15297,
        "Trots op Nederland": 13198,
        "Henk Krol List": 9264,
        "NLBeter": 8657,
        "List 30": 8277,
        "Libertarian Party": 5546,
        "OpRecht [nl]": 5449,
        "Jezus Leeft": 5015,
        "The Party Party": 3744,
        "Ubuntu Connected Front": 1880,
        "Free and Social Netherlands [nl]": 942,
        "Party of Unity [nl]": 804,
        "We are the Netherlands [nl]": 553,
        "Party for the Republic": 255,
        "Modern Netherlands [nl]": 245,
        "The Greens": 119
    }
    
    votes_series = pd.Series(test_data)
    total_votes = votes_series.sum()
    proportions = votes_series / total_votes

    # Reshape into a single-row DataFrame, like our simulation results
    sims_df = pd.DataFrame([proportions])

    # Run the allocation process
    initial_seats = calculate_hare_seats(sims_df, NUM_SEATS)
    final_seats = calculate_d_hondt_seats(sims_df, initial_seats, total_votes, NUM_SEATS)

    print("\n--- Test Results: Initial Seats (Hare) ---")
    print(initial_seats.iloc[0].to_string())
    print(f"Total initial seats: {initial_seats.iloc[0].sum()}")

    print("\n--- Test Results: Final Seats (D'Hondt) ---")
    print(final_seats.iloc[0].to_string())
    print(f"Total final seats: {final_seats.iloc[0].sum()}")
    print("--- End of Seat Allocation Test ---")


# --- Step 18: Write Timestamp to Preference Tab ---
def write_timestamp_to_preference_tab(sheet_key, tab_name):
    """Writes current timestamp to the preference tab and pořadí_aktuální_aging tab, column E, row 2."""
    print(f"\n--- Step 18: Writing Timestamp to Tabs ---")
    
    try:
        gc = gspread.service_account()
        sh = gc.open_by_key(sheet_key)
        
        # Get current time in GMT
        current_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S GMT")
        
        # Write to preference tab
        try:
            ws = sh.worksheet(tab_name)
            ws.update([[current_time]], range_name='E2')
            print(f"✓ Successfully wrote timestamp to {tab_name} tab: {current_time}")
        except Exception as e:
            print(f"❌ Error writing timestamp to {tab_name} tab: {e}")
        
        # Write to pořadí_aktuální_aging tab
        try:
            ws_ranks = sh.worksheet('pořadí_aktuální_aging')
            ws_ranks.update([[current_time]], range_name='E2')
            print(f"✓ Successfully wrote timestamp to pořadí_aktuální_aging tab: {current_time}")
        except Exception as e:
            print(f"❌ Error writing timestamp to pořadí_aktuální_aging tab: {e}")
        
    except Exception as e:
        print(f"❌ Error connecting to Google Sheets: {e}")


# --- Execution ---
if __name__ == "__main__":
    # Steps 1 & 2
    preference_df = load_data(SHEET_KEY, TAB_NAME)
    if preference_df is not None:
        processed_df, poll_date, total_valid_votes, others_p = process_data(preference_df.copy())
        print("\n--- Processed Data Check ---")
        print(processed_df.head())

        # Step 3
        print("\n--- Step 3: Aging Calculation Check ---")
        election_day = datetime.date.fromisoformat(ELECTION_DATE)
        aging = aging_coeff(poll_date, election_day)
        print(f"Aging coefficient calculated: {aging:.4f}")

        # Step 5
        sims, sims_aging = run_simulation(processed_df, SAMPLE_SIM, SAMPLE_N, aging)
        print("\n--- Standard Simulation Results Check ---")
        print("Standard Sims Head:")
        print(sims.head())
        print("\nAged Standard Sims Head:")
        print(sims_aging.head())

        # --- Step 7.5: Normalize Simulation Results ---
        # This is the critical fix: scale simulated party proportions to sum to (1 - others_p)
        print(f"\n--- Normalizing simulation results to sum to {1.0 - others_p:.4f} ---")
        target_sum = 1.0 - others_p
        sims_aging = sims_aging.div(sims_aging.sum(axis=1), axis=0).multiply(target_sum)
        print("Standard simulation results normalized.")

        # --- Step 8 & 9: Calculate final seats for standard simulation ---
        hare_seats_std_df = calculate_hare_seats(sims_aging, NUM_SEATS)
        final_seats_std_df = calculate_d_hondt_seats(sims_aging, hare_seats_std_df, total_valid_votes, NUM_SEATS)
        print("\n--- Final Seats (Standard Sim) ---")
        print(final_seats_std_df.head())

        print("\n--- Control Sum for Final Seats (Standard Sim) ---")
        seat_sums_std = final_seats_std_df.sum(axis=1)
        print("Distribution of total seats per simulation:")
        print(seat_sums_std.value_counts())

        # --- Step 10: Calculate Rank Probabilities (Standard Sim) ---
        calculate_and_print_rank_probabilities(final_seats_std_df, "Standard")

        # --- Step 11: Calculate Interval Probabilities (Standard Sim) ---
        calculate_and_print_interval_probabilities(final_seats_std_df, sims_aging, "Standard", "yes-no-cov")

        # --- Step 12: Calculate Top 5 Rank Probabilities (Standard Sim) ---
        calculate_and_print_top5_rank_probabilities(final_seats_std_df, "Standard")

        # --- Step 13: Write Ranks to Google Sheet (Standard Sim) ---
        write_ranks_to_sheet(final_seats_std_df, processed_df, SHEET_KEY, RANKS_NO_COV_TAB_NAME)


        # Steps 6 & 7
        correlation_df = load_correlation_data(SHEET_KEY, CORR_TAB_NAME)
        if correlation_df is not None:
            sims_cov, sims_aging_cov = run_simulation_with_covariance(processed_df, correlation_df, SAMPLE_SIM, aging)
            print("\n--- Covariance Simulation Results Check ---")
            print("Covariance Sims Head:")
            print(sims_cov.head())
            print("\nAged Covariance Sims Head:")
            print(sims_aging_cov.head())

            # Normalize the covariance simulation results as well
            sims_aging_cov = sims_aging_cov.div(sims_aging_cov.sum(axis=1), axis=0).multiply(target_sum)
            print("Covariance simulation results normalized.")

            # --- Step 8 & 9: Calculate final seats for covariance simulation ---
            hare_seats_cov_df = calculate_hare_seats(sims_aging_cov, NUM_SEATS)
            final_seats_cov_df = calculate_d_hondt_seats(sims_aging_cov, hare_seats_cov_df, total_valid_votes, NUM_SEATS)
            print("\n--- Final Seats (Covariance Sim) ---")
            print(final_seats_cov_df.head())

            print("\n--- Control Sum for Final Seats (Covariance Sim) ---")
            seat_sums_cov = final_seats_cov_df.sum(axis=1)
            print("Distribution of total seats per simulation:")
            print(seat_sums_cov.value_counts())

            # --- Step 10: Calculate Rank Probabilities (Covariance Sim) ---
            calculate_and_print_rank_probabilities(final_seats_cov_df, "Covariance")

            # --- Step 11: Calculate Interval Probabilities (Covariance Sim) ---
            calculate_and_print_interval_probabilities(final_seats_cov_df, sims_aging_cov, "Covariance", "yes")

            # --- Step 12: Calculate Top 5 Rank Probabilities (Covariance Sim) ---
            calculate_and_print_top5_rank_probabilities(final_seats_cov_df, "Covariance")

            # --- Step 13: Write Ranks to Google Sheet (Covariance Sim) ---
            write_ranks_to_sheet(final_seats_cov_df, processed_df, SHEET_KEY, RANKS_TAB_NAME)

            # --- Step 14: Write Top 5 Ranks to Google Sheet (Both Sims) ---
            write_top5_ranks_to_sheet(final_seats_cov_df, final_seats_std_df, processed_df, SHEET_KEY, TOP5_TAB_NAME)

            # --- Step 15: Write Summary Stats (Both Sims) ---
            calculate_and_write_summary_stats(final_seats_cov_df, hare_seats_cov_df, final_seats_std_df, SHEET_KEY, TAB_NAME)

            # --- Step 17: Calculate and Write Duels (Both Sims) ---
            calculate_and_write_duels(final_seats_cov_df, final_seats_std_df, SHEET_KEY, DUELS_TAB_NAME)

            # --- Step 18: Write Current Time to Preference Tab ---
            write_timestamp_to_preference_tab(SHEET_KEY, TAB_NAME)

        # Run the seat allocation test (now disabled)
        # test_seat_allocation()