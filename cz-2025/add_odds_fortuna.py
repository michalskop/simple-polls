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


