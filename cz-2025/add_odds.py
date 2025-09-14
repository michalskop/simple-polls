"""Add odds to GSheet."""

import datetime
import gspread
import pandas as pd
import re

# source sheet
sheetkey = "1es2J0O_Ig7RfnVHG3bHmX8SBjlMvrPwn4s1imYkxbwg"
path = "cz-2025/"

# load data from GSheet
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)

# Mapping from our names
# tipsport
# ANO	SPOLU	STAN	SPD	Piráti	Stačilo!	Motoristé	Přísaha
mappingt = {
  "ANO": "ANO 2011",
  "SPOLU": "SPOLU",
  "STAN": "STAN",
  "SPD": "SPD",
  "Piráti": "Piráti",
  "Stačilo!": "STAČILO!",
  "Motoristé": "Motoristé sobě",
  "Přísaha": "Přísaha",
}
# candidates are keys
candidates = [k for k in mappingt.keys()]

# RANK
####################
ws = sh.worksheet('pořadí_aktuální_aging_cov')
dfr = pd.DataFrame(ws.get_all_records())

urlt = "https://github.com/michalskop/tipsport.cz/raw/refs/heads/main/v4/data/6521550.csv"
dft = pd.read_csv(urlt, encoding="utf-8")

# Filter for the newest data only (max date)
if 'date' in dft.columns:
    max_date = dft['date'].max()
    dft = dft[dft['date'] == max_date]
    print(f"Using data from date: {max_date}")
else:
    max_date = "No date column found"

# Create the odds structure
odds_data = {}

# Initialize the structure for each party
for party in candidates:
    odds_data[party] = {
        'Ano': {},  # For "Ano" outcomes
        'Ne': {}    # For "Ne" outcomes
    }

# Extract odds for each party and ranking position
for party in candidates:
    tipsport_name = mappingt[party]
    
    # 1st place (most votes)
    event_name_1st = f"Nejvíc hlasů získá {tipsport_name}"
    odds_1st_ano = dft[(dft['event_name'] == event_name_1st) & (dft['name'] == 'Ano')]['odd'].values
    odds_1st_ne = dft[(dft['event_name'] == event_name_1st) & (dft['name'] == 'Ne')]['odd'].values
    
    if len(odds_1st_ano) > 0:
        odds_data[party]['Ano'][1] = odds_1st_ano[0]
    if len(odds_1st_ne) > 0:
        odds_data[party]['Ne'][1] = odds_1st_ne[0]
    
    # 2nd place
    event_name_2nd = f"Umístění {tipsport_name} do 2. místa podle počtu hlasů"
    odds_2nd_ano = dft[(dft['event_name'] == event_name_2nd) & (dft['name'] == 'Ano')]['odd'].values
    odds_2nd_ne = dft[(dft['event_name'] == event_name_2nd) & (dft['name'] == 'Ne')]['odd'].values
    
    if len(odds_2nd_ano) > 0:
        odds_data[party]['Ano'][2] = odds_2nd_ano[0]
    if len(odds_2nd_ne) > 0:
        odds_data[party]['Ne'][2] = odds_2nd_ne[0]
    
    # 3rd place
    event_name_3rd = f"Umístění {tipsport_name} do 3. místa podle počtu hlasů"
    odds_3rd_ano = dft[(dft['event_name'] == event_name_3rd) & (dft['name'] == 'Ano')]['odd'].values
    odds_3rd_ne = dft[(dft['event_name'] == event_name_3rd) & (dft['name'] == 'Ne')]['odd'].values
    
    if len(odds_3rd_ano) > 0:
        odds_data[party]['Ano'][3] = odds_3rd_ano[0]
    if len(odds_3rd_ne) > 0:
        odds_data[party]['Ne'][3] = odds_3rd_ne[0]
    
    # 4th place
    event_name_4th = f"Umístění {tipsport_name} do 4. místa podle počtu hlasů"
    odds_4th_ano = dft[(dft['event_name'] == event_name_4th) & (dft['name'] == 'Ano')]['odd'].values
    odds_4th_ne = dft[(dft['event_name'] == event_name_4th) & (dft['name'] == 'Ne')]['odd'].values
    
    if len(odds_4th_ano) > 0:
        odds_data[party]['Ano'][4] = odds_4th_ano[0]
    if len(odds_4th_ne) > 0:
        odds_data[party]['Ne'][4] = odds_4th_ne[0]

# Print the extracted odds data
print("Extracted odds data:")
print("=" * 50)
for party in candidates:
    print(f"\n{party}:")
    print("  Ano:")
    for rank in [1, 2, 3, 4]:
        if rank in odds_data[party]['Ano']:
            print(f"    {rank}. místo: {odds_data[party]['Ano'][rank]}")
        else:
            print(f"    {rank}. místo: N/A")
    
    print("  Ne:")
    for rank in [1, 2, 3, 4]:
        if rank in odds_data[party]['Ne']:
            print(f"    {rank}. místo: {odds_data[party]['Ne'][rank]}")
        else:
            print(f"    {rank}. místo: N/A")

# Write to Google Sheets
try:
    # Get the existing worksheet
    wsw = sh.worksheet('pořadí_aktuální_aging_cov')
    
    # Prepare "Ano" data - just the odds values
    # B26 = ANO 1st odds, C26 = SPOLU 1st odds, etc.
    # B27 = ANO 2nd odds, C27 = SPOLU 2nd odds, etc.
    
    ano_data_rows = []
    
    # Create data rows for each rank
    for rank in [1, 2, 3, 4]:
        row = []
        for party in candidates:
            if rank in odds_data[party]['Ano']:
                row.append(odds_data[party]['Ano'][rank])
            else:
                row.append('')
        ano_data_rows.append(row)
    
    # Write "Ano" data starting at B25 (4 rows)
    wsw.update(values=ano_data_rows, range_name='B25')
    
    # Prepare "Ne" data - just the odds values
    ne_data_rows = []
    
    # Create data rows for each rank
    for rank in [1, 2, 3, 4]:
        row = []
        for party in candidates:
            if rank in odds_data[party]['Ne']:
                row.append(odds_data[party]['Ne'][rank])
            else:
                row.append('')
        ne_data_rows.append(row)
    
    # Write "Ne" data starting at B44 (4 rows)
    wsw.update(values=ne_data_rows, range_name='B43')
    
    # Write the date to A24 and A42
    wsw.update(values=[[max_date]], range_name='A24')
    wsw.update(values=[[max_date]], range_name='A42')
    
    print("Successfully updated 'pořadí_aktuální_aging_cov' worksheet with odds data!")
    
except Exception as e:
    print(f"Error updating Google Sheets: {e}")
    print("Please check your Google Sheets permissions and try again")

##############################################################
# Extract percentage vote data for each party
print("\n" + "="*60)
print("EXTRACTING PERCENTAGE VOTE DATA:")
print("="*60)

# Get the limits from Google Sheet
ws_percentages = sh.worksheet('pravděpodobnosti_aktuální_aging_cov')
limits_data = ws_percentages.get('A2:A100')  # Get all values from column A starting from row 2
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
    tipsport_name = mappingt[party]
    event_name = f"{tipsport_name} - počet hlasů v procentech"
    
    # Filter data for this party's percentage events
    party_data = dft[dft['event_name'] == event_name]
    
    if len(party_data) == 0:
        print(f"No data found for {party} ({tipsport_name})")
        continue
    
    print(f"\n{party} ({tipsport_name}) - Found {len(party_data)} records:")
    
    party_odds = {}
    
    for _, row in party_data.iterrows():
        name = row['name']
        odd = row['odd']
        
        print(f"  {name}: {odd}")
        
        # Parse the percentage threshold
        if "Méně než" in name:
            # Extract number from "Méně než 33.01%"
            match = re.search(r'(\d+\.?\d*)%', name)
            if match:
                threshold = float(match.group(1))
                party_odds[f"less_than_{threshold}"] = odd
        elif "a více" in name:
            # Extract number from "33.01% a více"
            match = re.search(r'(\d+\.?\d*)%', name)
            if match:
                threshold = float(match.group(1))
                party_odds[f"more_than_{threshold}"] = odd
    
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
    expected_threshold = limit + 0.01
    
    if expected_threshold in all_thresholds and expected_threshold not in used_thresholds:
        matches.append((limit, expected_threshold))
        used_thresholds.add(expected_threshold)
        print(f"Matched Google Sheet limit {limit}% with data threshold {expected_threshold}%")
    else:
        # Fallback to closest match if exact 0.01 difference not found
        best_match = None
        best_diff = float('inf')
        
        for threshold in all_thresholds:
            if threshold in used_thresholds:
                continue  # Skip already used thresholds
            
            diff = abs(limit - threshold)
            if diff < best_diff and diff < 0.5:  # Allow up to 0.5% difference
                best_match = threshold
                best_diff = diff
        
        if best_match:
            matches.append((limit, best_match))
            used_thresholds.add(best_match)
            print(f"Matched Google Sheet limit {limit}% with data threshold {best_match}% (fallback)")

print(f"\nTotal matches found: {len(matches)}")
print(f"Used thresholds: {sorted(used_thresholds)}")

# Create a mapping from Google Sheet limits to data thresholds
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
    
    # Calculate the range: V2 to the last column for all limits
    # V is column 22, so for 8 parties: V(22), W(23), X(24), Y(25), Z(26), AA(27), AB(28), AC(29)
    start_col_num = 22  # V
    end_col_num = start_col_num + len(candidates) - 1  # 22 + 8 - 1 = 29

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
    range_name = f'V2:{end_col}{end_row}'
    
    print(f"Writing data to range: {range_name}")
    print(f"Data shape: {len(write_data)} limits x {len(candidates)} parties")
    
    # Write the data
    ws_percentages.update(values=write_data, range_name=range_name)
    
    print("Successfully updated 'pravděpodobnosti_aktuální_aging_cov' worksheet with percentage data!")
    
except Exception as e:
    print(f"Error updating Google Sheets: {e}")
    print("Please check your Google Sheets permissions and try again")
