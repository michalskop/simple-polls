"""Add odds to GSheet."""

import datetime
import gspread
import pandas as pd
import re
import time

# source sheet
sheetkey = "1es2J0O_Ig7RfnVHG3bHmX8SBjlMvrPwn4s1imYkxbwg"
path = "cz-2025/"

# load data from GSheet
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)

# Mapping from our names
mappingf = {
  "ANO": "ANO 2011",
  "SPOLU": "SPOLU",
  "STAN": "STAN",
  "SPD": "SPD",
  "Piráti": "Piráti",
  "Stačilo!": "Stačilo!",
  "Motoristé": "Motoristé sobě",
  "Přísaha": "Přísaha",
}

# candidates are keys
candidates = [k for k in mappingf.keys()]

# RANK
ws = sh.worksheet('pořadí_aktuální_aging_cov')
dfr = pd.DataFrame(ws.get_all_records())

urlf = "https://github.com/michalskop/ifortuna.cz/raw/master/data/MCZ5158.v1.2.csv"
dff = pd.read_csv(urlf, encoding="utf-8")

# Filter for the newest data only (max date)
if 'date' in dff.columns:
    max_date = dff['date'].max()
    dff = dff[dff['date'] == max_date]
    print(f"Using data from date: {max_date}")
else:
    max_date = "No date column found"

# Clean formatted numbers (remove all spaces and convert to float)
odds_columns = ['odds1', 'odds2', 'odds3', 'odds4', 'odds5', 'odds6']
for col in odds_columns:
    if col in dff.columns:
        # Remove all spaces and convert to float
        dff[col] = dff[col].astype(str).str.replace(r'\s+', '', regex=True).astype(float)

print("Sample of cleaned data:")
print(dff[['event_name'] + odds_columns].head())

# Reorder dff to have parties as columns and positions as rows
def reorder_odds_dataframe(dff, mappingf):
    """
    Reorder DataFrame to have parties as columns and positions as rows.
    
    Args:
        dff: DataFrame with columns date, event_info_number, event_name, event_link, odds, datum, odds1, odds2, odds3, odds4, odds5, odds6
        mappingf: Dictionary mapping our party names to Fortuna party names
    
    Returns:
        DataFrame with parties as columns and positions as rows
    """
    # Create a new DataFrame with parties as columns
    parties = list(mappingf.keys())
    positions = ['1st', '2nd', '3rd', '4th', '5th', '6th']
    
    # Initialize the result DataFrame
    result_df = pd.DataFrame(index=positions, columns=parties)
    
    # Process each row in dff
    for _, row in dff.iterrows():
        party_name = row['event_name']
        
        # Find which party this corresponds to in our mapping
        mapped_party = None
        for our_party, fortuna_party in mappingf.items():
            if fortuna_party == party_name:
                mapped_party = our_party
                break
        
        if mapped_party is None:
            continue  # Skip if party not in our mapping
        
        # Extract odds for each position
        for i, position in enumerate(positions, 1):
            odds_col = f'odds{i}'
            odds_value = row[odds_col]
            
            # Handle missing values (-1.00) or NaN
            if pd.isna(odds_value) or odds_value == -1.00:
                result_df.loc[position, mapped_party] = None
            else:
                # Ensure the value is properly formatted as a number
                if isinstance(odds_value, str):
                    # Clean any remaining spaces and convert to float
                    odds_value = float(str(odds_value).replace(' ', ''))
                result_df.loc[position, mapped_party] = odds_value
    
    return result_df

# Apply the reordering
reordered_df = reorder_odds_dataframe(dff, mappingf)

# Display the result
print("Reordered DataFrame:")
print(reordered_df)
print("=" * 50)

# Write to Google Sheets
ws = sh.worksheet('pořadí_aktuální_aging_cov')
ws.update(reordered_df.values.tolist(), range_name='B74')

# Write to Google Sheets last date from dff
ws.update(values=[[max_date]], range_name='A73')

# wait 10 seconds to avoid rate limit
time.sleep(10)

print("Successfully updated 'pořadí_aktuální_aging_cov' worksheet with odds data!")

##############################################################
# Extract percentage vote data for each party
print("\n" + "="*60)
print("EXTRACTING PERCENTAGE VOTE DATA:")
print("="*60)

# Load percentage data from Fortuna
urlf_percent = "https://raw.githubusercontent.com/michalskop/ifortuna.cz/refs/heads/master/data/MCZ51650.v2-1.csv"
dff_percent = pd.read_csv(urlf_percent, encoding="utf-8")

# Filter for the newest data only (max date)
if 'date' in dff_percent.columns:
    max_date_percent = dff_percent['date'].max()
    dff_percent = dff_percent[dff_percent['date'] == max_date_percent]
    print(f"Using percentage data from date: {max_date_percent}")
else:
    max_date_percent = "No date column found"

# Clean formatted numbers (remove spaces and convert to float)
for col in ['odd1', 'odd2']:
    if col in dff_percent.columns:
        dff_percent[col] = dff_percent[col].astype(str).str.replace(r'\s+', '', regex=True).astype(float)

# Get the limits from Google Sheet
ws_percentages = sh.worksheet('pravděpodobnosti_aktuální_aging_cov')
limits_data = ws_percentages.get('A2:A106')  # Get all values from column A starting from row 2
limits = []
for row in limits_data:
    if row and row[0]:  # Check if row exists and has data
        try:
            # Convert to float, handling both "33.5" and "33.51%" formats
            limit_str = str(row[0]).replace('%', '')
            limit_val = float(limit_str)
            limits.append(limit_val)
        except ValueError:
            print(f"Could not parse limit: {row[0]}")
            continue

print(f"Found {len(limits)} limits: {limits[:10]}...")  # Show first 10 limits

# Extract percentage vote data for each party
percentage_data = {}

for party in candidates:
    fortuna_name = mappingf[party]
    
    # Filter data for this party's percentage events
    party_data = dff_percent[dff_percent['event_name'] == fortuna_name]
    
    if len(party_data) == 0:
        print(f"No data found for {party} ({fortuna_name})")
        continue
    
    print(f"\n{party} ({fortuna_name}) - Found {len(party_data)} records:")
    
    party_odds = {}
    
    for _, row in party_data.iterrows():
        header1 = row['header1']
        header2 = row['header2']
        odd1 = row['odd1']
        odd2 = row['odd2']
        
        print(f"  {header1}: {odd1}")
        print(f"  {header2}: {odd2}")
        
        # Parse the percentage threshold from header1 (Méně než X.XX)
        if "Méně než" in header1:
            # Extract number from "Méně než 30.01"
            match = re.search(r'(\d+\.?\d*)', header1)
            if match:
                threshold = float(match.group(1))
                # odd1 is for "less_than" (header1)
                if odd1 != -1.00:  # Only add if not missing value
                    party_odds[f"less_than_{threshold}"] = odd1
        
        # Parse the percentage threshold from header2 (Více než X.XX)
        if "Více než" in header2:
            # Extract number from "Více než 30.01"
            match = re.search(r'(\d+\.?\d*)', header2)
            if match:
                threshold = float(match.group(1))
                # odd2 is for "more_than" (header2)
                if odd2 != -1.00:  # Only add if not missing value
                    party_odds[f"more_than_{threshold}"] = odd2
    
    percentage_data[party] = party_odds

# Print the extracted percentage data
print("\n" + "="*60)
print("EXTRACTED PERCENTAGE DATA:")
print("="*60)
for party in candidates:
    if party in percentage_data:
        print(f"\n{party}:")
        for key, value in percentage_data[party].items():
            print(f"  {key}: {value}")
    else:
        print(f"\n{party}: No data found")

# Find all unique thresholds from the data
all_thresholds = set()
for party_data in percentage_data.values():
    for key in party_data.keys():
        if key.startswith('more_than_'):
            threshold = float(key.replace('more_than_', ''))
            all_thresholds.add(threshold)
        elif key.startswith('less_than_'):
            threshold = float(key.replace('less_than_', ''))
            all_thresholds.add(threshold)

all_thresholds = sorted(list(all_thresholds))
print(f"\nAll thresholds found in data: {all_thresholds}")

# Compare with Google Sheet limits
print(f"\nGoogle Sheet limits: {limits[:10]}...")
print(f"Data thresholds: {all_thresholds[:10]}...")

# Check for exact matches with 0.01 difference
matches = []
used_thresholds = set()  # Keep track of used thresholds

for limit in limits:
    # Check if there's a data threshold that's exactly 0.01 bigger
    target_threshold = limit + 0.01
    if target_threshold in all_thresholds and target_threshold not in used_thresholds:
        matches.append((limit, target_threshold))
        used_thresholds.add(target_threshold)
        print(f"Match: GS limit {limit}% -> Data threshold {target_threshold}%")
    else:
        # Also check for exact matches (in case some don't follow the 0.01 pattern)
        if limit in all_thresholds and limit not in used_thresholds:
            matches.append((limit, limit))
            used_thresholds.add(limit)
            print(f"Match: GS limit {limit}% -> Data threshold {limit}%")

print(f"\nFound {len(matches)} matches between Google Sheet limits and data thresholds")

# Create mapping from Google Sheet limits to data thresholds
limit_to_threshold = {}
for gs_limit, data_threshold in matches:
    limit_to_threshold[gs_limit] = data_threshold

# Prepare percentage data matrix for Google Sheet
print("\n" + "="*60)
print("PREPARING PERCENTAGE DATA FOR GOOGLE SHEET:")
print("="*60)

# Create a matrix: parties x limits
# For each party and each limit, find the appropriate odds
percentage_matrix = []

for party in candidates:
    party_row = [party]
    
    for limit in limits:
        # Check if we have a mapping for this limit
        if limit in limit_to_threshold:
            data_threshold = limit_to_threshold[limit]
            key = f"more_than_{data_threshold}"
            
            if party in percentage_data and key in percentage_data[party]:
                party_row.append(percentage_data[party][key])
            else:
                party_row.append('')  # Empty if no data
        else:
            party_row.append('')  # Empty if no mapping found
    
    percentage_matrix.append(party_row)

print("Percentage matrix (Party x Limits):")
print("Party", end="")
for i, limit in enumerate(limits[:10]):  # Show first 10 limits
    print(f"\t{limit}%", end="")
print()

for row in percentage_matrix:
    print(row[0], end="")
    for i, value in enumerate(row[1:11]):  # Show first 10 values
        print(f"\t{value}", end="")
    print()

print(f"\nMatrix size: {len(percentage_matrix)} parties x {len(limits)} limits")

# Count non-empty values
non_empty_count = sum(1 for row in percentage_matrix for value in row[1:] if value != '')
print(f"Non-empty values: {non_empty_count}")

# Show which limits have data
print("\nLimits with data:")
for i, limit in enumerate(limits):
    has_data = any(row[i+1] != '' for row in percentage_matrix)
    if has_data:
        print(f"  {limit}%: YES")
    else:
        print(f"  {limit}%: NO")

# Write percentage data to Google Sheet
print("\n" + "="*60)
print("WRITING PERCENTAGE DATA TO GOOGLE SHEET:")
print("="*60)

try:
    # Get the percentage worksheet
    ws_percentages = sh.worksheet('pravděpodobnosti_aktuální_aging_cov')
    
    # Prepare data for writing (limits as rows, parties as columns)
    # Start at V2, each limit gets one row
    write_data = []
    
    for i, limit in enumerate(limits):
        limit_row = []
        
        for party in candidates:
            # Check if we have a mapping for this limit
            if limit in limit_to_threshold:
                data_threshold = limit_to_threshold[limit]
                key = f"more_than_{data_threshold}"
                
                if party in percentage_data and key in percentage_data[party]:
                    limit_row.append(percentage_data[party][key])
                else:
                    limit_row.append('')  # Empty if no data
            else:
                limit_row.append('')  # Empty if no mapping found
        
        write_data.append(limit_row)
    
    # Calculate the range: BJ2 to the last column for all limits
    # BJ is column 62, so for 8 parties: BJ(62), BK(63), BL(64), BM(65), BN(66), BO(67), BP(68), BQ(69)
    start_col_num = 62  # BJ
    end_col_num = start_col_num + len(candidates) - 1  # 62 + 8 - 1 = 69

    # Convert column numbers to letters
    def num_to_col(num):
        result = ""
        while num > 0:
            num -= 1
            result = chr(65 + (num % 26)) + result
            num //= 26
        return result

    end_col = num_to_col(end_col_num)
    end_row = 2 + len(limits) - 1
    range_name = f'BJ2:{end_col}{end_row}'
    
    print(f"Writing data to range: {range_name}")
    print(f"Data shape: {len(write_data)} limits x {len(candidates)} parties")
    
    # Write the data
    ws_percentages.update(values=write_data, range_name=range_name)
    
    print("Successfully updated 'pravděpodobnosti_aktuální_aging_cov' worksheet with percentage data!")
    
except Exception as e:
    print(f"Error updating Google Sheets: {e}")
    print("Please check your Google Sheets permissions and try again")

# Write "less_than" percentage data to Google Sheet
print("\n" + "="*60)
print("WRITING LESS_THAN PERCENTAGE DATA TO GOOGLE SHEET:")
print("="*60)

try:
    # Get the percentage worksheet
    ws_percentages = sh.worksheet('pravděpodobnosti_aktuální_aging_cov')
    
    # Prepare "less_than" data for writing (limits as rows, parties as columns)
    # Start at AP2, each limit gets one row
    less_than_data = []
    
    for i, limit in enumerate(limits):
        limit_row = []
        
        for party in candidates:
            # Check if we have a mapping for this limit
            if limit in limit_to_threshold:
                data_threshold = limit_to_threshold[limit]
                key = f"less_than_{data_threshold}"
                
                if party in percentage_data and key in percentage_data[party]:
                    limit_row.append(percentage_data[party][key])
                else:
                    limit_row.append('')  # Empty if no data
            else:
                limit_row.append('')  # Empty if no mapping found
        
        less_than_data.append(limit_row)
    
    # Calculate the range: CD2 to the last column for all limits
    # CD is column 82, so for 8 parties: CD(82), CE(83), CF(84), CG(85), CH(86), CI(87), CJ(88), CK(89)
    start_col_num = 82  # CD
    end_col_num = start_col_num + len(candidates) - 1  # 82 + 8 - 1 = 89

    end_col = num_to_col(end_col_num)
    end_row = 2 + len(limits) - 1
    range_name = f'CD2:{end_col}{end_row}'
    
    print(f"Writing less_than data to range: {range_name}")
    print(f"Data shape: {len(less_than_data)} limits x {len(candidates)} parties")
    
    # Write the data
    ws_percentages.update(values=less_than_data, range_name=range_name)
    
    print("Successfully updated 'pravděpodobnosti_aktuální_aging_cov' worksheet with less_than percentage data!")
    
except Exception as e:
    print(f"Error updating Google Sheets: {e}")
    print("Please check your Google Sheets permissions and try again")

print("\n" + "="*60)
print("PERCENTAGE DATA EXTRACTION COMPLETED!")
print("="*60)

##############################################################
# Extract duel odds data
print("\n" + "="*60)
print("EXTRACTING DUEL ODDS DATA:")
print("="*60)

# Load duel data from Fortuna
urlf_duels = "https://raw.githubusercontent.com/michalskop/ifortuna.cz/refs/heads/master/data/MCZ51652.v1.2.csv"
dff_duels = pd.read_csv(urlf_duels, encoding="utf-8")

# Filter for the newest data only (max date)
if 'date' in dff_duels.columns:
    max_date_duels = dff_duels['date'].max()
    dff_duels = dff_duels[dff_duels['date'] == max_date_duels]
    print(f"Using duel data from date: {max_date_duels}")
else:
    max_date_duels = "No date column found"

# Clean formatted numbers (remove spaces and convert to float)
for col in ['odds1', 'odds2']:
    if col in dff_duels.columns:
        dff_duels[col] = dff_duels[col].astype(str).str.replace(r'\s+', '', regex=True).astype(float)

# Create duel odds matrix
duel_odds = {}

for i, party1 in enumerate(candidates):
    for j, party2 in enumerate(candidates):
        if i != j:  # Skip same party duels
            fortuna_name1 = mappingf[party1]
            fortuna_name2 = mappingf[party2]
            
            # Try both possible event name formats
            event_name1 = f"{fortuna_name1} - {fortuna_name2}"
            event_name2 = f"{fortuna_name2} - {fortuna_name1}"
            
            # Find the odds for party1 beating party2
            duel_data1 = dff_duels[dff_duels['event_name'] == event_name1]
            duel_data2 = dff_duels[dff_duels['event_name'] == event_name2]
            
            found_odds = None
            
            # Try first format (party1 - party2)
            if len(duel_data1) > 0:
                # odds1 is for the first party in the event name
                party1_odds = duel_data1['odds1'].values
                if len(party1_odds) > 0 and party1_odds[0] != -1.00:
                    found_odds = party1_odds[0]
                    print(f"{party1} vs {party2}: {found_odds} (format 1)")
            
            # Try second format (party2 - party1) if first didn't work
            if found_odds is None and len(duel_data2) > 0:
                # odds2 is for the second party in the event name, which is party1 in our case
                party1_odds = duel_data2['odds2'].values
                if len(party1_odds) > 0 and party1_odds[0] != -1.00:
                    found_odds = party1_odds[0]
                    print(f"{party1} vs {party2}: {found_odds} (format 2)")
            
            if found_odds is not None:
                duel_odds[(party1, party2)] = found_odds
            else:
                print(f"No odds found for {party1} vs {party2}")

# Create the duel matrix for Google Sheet
print("\n" + "="*60)
print("PREPARING DUEL MATRIX FOR GOOGLE SHEET:")
print("="*60)

duel_matrix = []

for party1 in candidates:
    row = []
    for party2 in candidates:
        if party1 == party2:
            # Diagonal: party vs itself (empty or 1.0)
            row.append('')
        else:
            # Get odds for party1 beating party2
            if (party1, party2) in duel_odds:
                row.append(duel_odds[(party1, party2)])
            else:
                row.append('')  # Empty if no data
    duel_matrix.append(row)

# Print the matrix
print("Duel matrix (Row beats Column):")
print("Party", end="")
for party in candidates:
    print(f"\t{party}", end="")
print()

for i, party in enumerate(candidates):
    print(party, end="")
    for j, value in enumerate(duel_matrix[i]):
        print(f"\t{value}", end="")
    print()

print(f"\nMatrix size: {len(duel_matrix)} x {len(duel_matrix[0]) if duel_matrix else 0}")

# Write duel data to Google Sheet
print("\n" + "="*60)
print("WRITING DUEL DATA TO GOOGLE SHEET:")
print("="*60)

try:
    # Get the duel worksheet
    ws_duels = sh.worksheet('duely_aging_cov')
    
    # Calculate the range: B26 to the last column for all parties
    # B is column 2, so for 8 parties: B(2), C(3), D(4), E(5), F(6), G(7), H(8), I(9)
    start_col_num = 2  # B
    end_col_num = start_col_num + len(candidates) - 1  # 2 + 8 - 1 = 9

    # Convert column numbers to letters
    def num_to_col(num):
        result = ""
        while num > 0:
            num -= 1
            result = chr(65 + (num % 26)) + result
            num //= 26
        return result

    end_col = num_to_col(end_col_num)
    end_row = 47 + len(candidates) - 1  # 47 + 8 - 1 = 54
    range_name = f'B47:{end_col}{end_row}'
    
    print(f"Writing duel data to range: {range_name}")
    print(f"Data shape: {len(duel_matrix)} parties x {len(duel_matrix[0]) if duel_matrix else 0} parties")
    
    # Write the data
    ws_duels.update(values=duel_matrix, range_name=range_name)
    
    print("Successfully updated 'duely_aging_cov' worksheet with duel data!")
    
except Exception as e:
    print(f"Error updating Google Sheets with duel data: {e}")
    print("Please check your Google Sheets permissions and try again")

print("\n" + "="*60)
print("DUEL DATA EXTRACTION COMPLETED!")
print("="*60)


