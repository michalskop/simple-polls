"""Reading the markets for Bucuresti mayor 2025 elections."""

from py_clob_client.client import ClobClient
import gspread
import pandas as pd
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv(dotenv_path=".env")

# Connect to Polymarket
host: str = "https://clob.polymarket.com"
key: str = os.getenv('POLYMARKET_PRIVATE_KEY')
chain_id: int = 137
POLYMARKET_PROXY_ADDRESS: str = os.getenv('POLYMARKET_PROXY_ADDRESS')

# Initialize client
client = ClobClient(host, key=key, chain_id=chain_id, signature_type=2, funder=POLYMARKET_PROXY_ADDRESS)
creds = client.derive_api_key()
client.set_api_creds(creds)

# Load token IDs from CSV
csv_path = "pt-2026/pt_victory_token_ids.csv"
df_tokens = pd.read_csv(csv_path)

print("Loaded token data:")
print(df_tokens.head())

# Google Sheet configuration
sheetkey = "1hhRprFQjeQAFp435YUFBjqKCzxFp1q85J7TWAaWZ2JE"
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)
sheet = sh.worksheet("pořadí_aktuální_aging_cov")

# Get party order from Google Sheets (B1 to P1)
party_order = sheet.get('B1:I1')[0]  # Get first row, B to I
print(f"Party order from Google Sheets: {party_order}")

# Get all unique parties from the data
all_parties = df_tokens['name'].unique()
print(f"All parties in data: {sorted(all_parties)}")

# Get all unique ranks
all_ranks = sorted(df_tokens['rank'].unique())
print(f"All ranks: {all_ranks}")

def get_extremes(orderbook):
    """Get bid and ask prices from orderbook."""
    bids = [float(bid.price) for bid in orderbook.bids]
    asks = [float(ask.price) for ask in orderbook.asks]
    return max(bids, default=0), min(asks, default=1)

# Collect all orderbook data
print("\nFetching orderbook data...")
orderbook_data = {}

for _, row in df_tokens.iterrows():
    party = row['name']
    rank = row['rank']
    yes_token_id = row['yes_token_id']
    no_token_id = row['no_token_id']
    
    print(f"Fetching data for {party} rank {rank}...")
    
    # Get YES token data
    try:
        yes_orderbook = client.get_order_book(yes_token_id)
        yes_bid, yes_ask = get_extremes(yes_orderbook)
    except Exception as e:
        print(f"Error fetching YES data for {party} rank {rank}: {e}")
        yes_bid, yes_ask = 0, 1
    
    # NO values are derived from YES token data: 1 - max(bid)
    no_price = 1 - yes_bid if yes_bid > 0 else 0
    
    # Store the data
    key = f"{party}_{rank}"
    orderbook_data[key] = {
        'party': party,
        'rank': rank,
        'yes_bid': yes_bid,
        'yes_ask': yes_ask,
        'no_price': no_price
    }
    
    # wait a bit
    time.sleep(0.5)

print(f"\nCollected data for {len(orderbook_data)} markets")

# Prepare data for writing
print("\nPreparing data for Google Sheets...")

# Create matrices for YES and NO data
# Rows: ranks (1, 2, 12) - 12 is the 1st or 2nd place
# Columns: parties (in order from Google Sheets)

yes_data = []
no_data = []

for rank in all_ranks:
    yes_row = []
    no_row = []
    
    for party in party_order:
        if party:  # Skip empty cells
            # Look for data for this party and rank
            key = f"{party}_{rank}"
            if key in orderbook_data:
                data = orderbook_data[key]
                yes_price = data['yes_ask']  # Use ask price for YES
                no_price = data['no_price']  # Use calculated NO price (1 - max_bid)
                yes_row.append(yes_price)
                no_row.append(no_price)
                print(f"  {party} rank {rank}: YES={yes_price:.4f}, NO={no_price:.4f}")
            else:
                yes_row.append('')  # No data available
                no_row.append('')
                print(f"  {party} rank {rank}: No data")
        else:
            yes_row.append('')
            no_row.append('')
    
    yes_data.append(yes_row)
    no_data.append(no_row)

print(f"\nYES data matrix ({len(yes_data)} rows x {len(party_order)} columns):")
for i, row in enumerate(yes_data):
    print(f"  Rank {all_ranks[i]}: {row}")

print(f"\nNO data matrix ({len(no_data)} rows x {len(party_order)} columns):")
for i, row in enumerate(no_data):
    print(f"  Rank {all_ranks[i]}: {row}")

# Write to Google Sheets
print("\nWriting to Google Sheets...")

try:
    # Write current timestamp to B26
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Writing timestamp to B26: {current_time}")
    sheet.update([[current_time]], range_name='B26')
    print("✓ Timestamp written successfully")
    
    # Write YES data starting at B28
    yes_range = f"B28:{chr(ord('B') + len(party_order) - 1)}{28 + len(all_ranks) - 1}"
    print(f"Writing YES data to range: {yes_range}")
    sheet.update(yes_data, range_name=yes_range)
    print("✓ YES data written successfully")
    
    # Write NO data starting at B38
    no_range = f"B38:{chr(ord('B') + len(party_order) - 1)}{38 + len(all_ranks) - 1}"
    print(f"Writing NO data to range: {no_range}")
    sheet.update(no_data, range_name=no_range)
    print("✓ NO data written successfully")
    
    print("\n🎉 All data written successfully to Google Sheets!")
    
except Exception as e:
    print(f"❌ Error writing to Google Sheets: {e}")
    print("Please check your Google Sheets permissions and try again")

print(f"\nSummary:")
print(f"- Processed {len(orderbook_data)} markets")
print(f"- {len(all_ranks)} ranks: {all_ranks}")
print(f"- {len([p for p in party_order if p])} parties: {[p for p in party_order if p]}")
print(f"- YES data written to rows 28-{28 + len(all_ranks) - 1}")
print(f"- NO data written to rows 38-{38 + len(all_ranks) - 1}")

# --- Functionality starts here: pravděpodobnosti_aktuální_aging_cov ---

print("\nStarting new functionality: Writing prices based on pt_token_ids.csv")

prob_sheet = sh.worksheet("pravděpodobnosti_aktuální_aging_cov")

# Load the correct token IDs CSV
pt_csv_path = "pt-2026/pt_token_ids.csv"
df_pt_tokens = pd.read_csv(pt_csv_path)

print("Loaded pt_token_ids.csv data:")
print(df_pt_tokens.head())

# Helper function to convert column number to letter (1 = A, 27 = AA, 31 = AE, etc.)
def num_to_col(num):
    """Convert column number to Excel column letter(s)."""
    result = ""
    while num > 0:
        num -= 1
        result = chr(65 + (num % 26)) + result
        num //= 26
    return result

# Starting column is AE (column 31)
start_col_num = 31  # AE
# Get headers from the sheet for name matching (AE1:AL1 or similar range)
# Assuming we have up to 5 candidates for YES and 5 for NO
headers = prob_sheet.get(f'{num_to_col(start_col_num)}1:{num_to_col(start_col_num + 9)}1')[0]
yes_headers = headers[:5]  # First 5 columns for YES
no_headers = headers[5:10]  # Next 5 columns for NO

# Create a mapping from name to column letter
yes_name_to_col = {}
for i, name in enumerate(yes_headers):
    if name:  # Skip empty headers
        col_letter = num_to_col(start_col_num + i)
        yes_name_to_col[name] = col_letter

no_name_to_col = {}
for i, name in enumerate(no_headers):
    if name:  # Skip empty headers
        col_letter = num_to_col(start_col_num + 5 + i)
        no_name_to_col[name] = col_letter

print(f"YES column mapping: {yes_name_to_col}")
print(f"NO column mapping: {no_name_to_col}")

# Find columns with "lower" and "upper" headers
header_row = prob_sheet.get('1:1')[0]  # Get first row
lower_col_idx = None
upper_col_idx = None

for i, header in enumerate(header_row):
    if header and 'lower' in str(header).lower():
        lower_col_idx = i
        lower_col_letter = num_to_col(i + 1)  # Convert to 1-based column number
        print(f"Found 'lower' column at index {i} (column {lower_col_letter})")
    if header and 'upper' in str(header).lower():
        upper_col_idx = i
        upper_col_letter = num_to_col(i + 1)  # Convert to 1-based column number
        print(f"Found 'upper' column at index {i} (column {upper_col_letter})")

if lower_col_idx is None or upper_col_idx is None:
    print("❌ Error: Could not find 'lower' and/or 'upper' columns in the sheet header")
    print("Available headers:", [h for h in header_row if h])
    exit(1)

# Get lo/hi ranges from the sheet using the found columns
lower_col_letter = num_to_col(lower_col_idx + 1)
upper_col_letter = num_to_col(upper_col_idx + 1)

# Read columns separately in case they're not adjacent
lower_values = prob_sheet.get(f'{lower_col_letter}2:{lower_col_letter}50')  # Assuming max 50 rows
upper_values = prob_sheet.get(f'{upper_col_letter}2:{upper_col_letter}50')  # Assuming max 50 rows

# Combine into pairs
sheet_ranges = []
max_rows = max(len(lower_values), len(upper_values))
for i in range(max_rows):
    lower_val = lower_values[i][0] if i < len(lower_values) and len(lower_values[i]) > 0 else None
    upper_val = upper_values[i][0] if i < len(upper_values) and len(upper_values[i]) > 0 else None
    sheet_ranges.append([lower_val, upper_val])

# Debug: Print first few rows to understand the format
print(f"\nDebug: First 5 rows from columns {lower_col_letter} (lower) and {upper_col_letter} (upper):")
for i, sheet_row in enumerate(sheet_ranges[:5]):
    lower_val = sheet_row[0] if len(sheet_row) > 0 else 'N/A'
    upper_val = sheet_row[1] if len(sheet_row) > 1 else 'N/A'
    print(f"  Row {i+2}: lower={lower_val}, upper={upper_val}")

# Prepare batch update requests
yes_price_updates = []
no_price_updates = []

print("\nFetching prices for markets in pt_token_ids.csv...")

for index, row in df_pt_tokens.iterrows():
    name = row['name']
    lo = row['lo']
    hi = row['hi']
    yes_token_id = row['yes_token_id']
    no_token_id = row['no_token_id']

    # Find the corresponding row in the Google Sheet
    target_row_index = -1
    for i, sheet_row in enumerate(sheet_ranges):
        try:
            if len(sheet_row) >= 2:
                sheet_lo = int(float(sheet_row[0])) if sheet_row[0] else None
                sheet_hi = int(float(sheet_row[1])) if sheet_row[1] else None
                if sheet_lo == lo and sheet_hi == hi:
                    target_row_index = i + 2  # 1-based index, and data starts from row 2
                    break
        except (ValueError, IndexError, TypeError):
            continue

    if target_row_index == -1:
        print(f"Warning: No matching row found for {name} with range {lo}-{hi}")
        continue

    # Fetch YES price
    yes_price = ''
    try:
        orderbook = client.get_order_book(yes_token_id)
        _, yes_ask = get_extremes(orderbook)
        if yes_ask < 1:
            yes_price = yes_ask
    except Exception as e:
        print(f"Could not fetch YES price for {name} ({lo}-{hi}): {e}")

    # Fetch NO price (derived from YES token: 1 - max(bid))
    no_price = ''
    try:
        orderbook = client.get_order_book(yes_token_id)
        yes_bid, _ = get_extremes(orderbook)
        if yes_bid > 0:
            no_price = 1 - yes_bid
    except Exception as e:
        print(f"Could not fetch NO price for {name} ({lo}-{hi}): {e}")

    # Find column and prepare update
    if name in yes_name_to_col and yes_price:
        col = yes_name_to_col[name]
        cell = f"{col}{target_row_index}"
        yes_price_updates.append({'range': cell, 'values': [[yes_price]]})
        print(f"  Prepared YES update for {name} ({lo}-{hi}) at {cell}: {yes_price:.4f}")

    # For NO prices, check if name is in mapping, or use the same index as YES if available
    if no_price:
        if name in no_name_to_col:
            col = no_name_to_col[name]
            cell = f"{col}{target_row_index}"
            no_price_updates.append({'range': cell, 'values': [[no_price]]})
            print(f"  Prepared NO update for {name} ({lo}-{hi}) at {cell}: {no_price:.4f}")
        elif name in yes_name_to_col:
            # If NO column doesn't have the name, try to find corresponding NO column
            # by using the same index offset (YES starts at AE=31, NO starts at AK=37 which is +6)
            yes_col_num = start_col_num + list(yes_name_to_col.keys()).index(name)
            no_col_letter = num_to_col(start_col_num + 6 + list(yes_name_to_col.keys()).index(name))
            cell = f"{no_col_letter}{target_row_index}"
            no_price_updates.append({'range': cell, 'values': [[no_price]]})
            print(f"  Prepared NO update for {name} ({lo}-{hi}) at {cell}: {no_price:.4f} (using offset)")
        else:
            print(f"  Warning: Could not find NO column for {name} ({lo}-{hi}), NO price: {no_price:.4f}")

    time.sleep(0.5)  # Be respectful to the API

# Execute batch updates
print("\nExecuting batch updates to Google Sheets...")

try:
    if yes_price_updates:
        prob_sheet.batch_update(yes_price_updates)
        print(f"✓ Successfully updated {len(yes_price_updates)} YES prices.")
    else:
        print("No YES prices to update.")

    if no_price_updates:
        prob_sheet.batch_update(no_price_updates)
        print(f"✓ Successfully updated {len(no_price_updates)} NO prices.")
    else:
        print("No NO prices to update.")

    print("\n🎉 New functionality executed successfully!")

except Exception as e:
    print(f"❌ Error during batch update: {e}")


# --- New functionality for victory margin tokens ---

print("\n--- Starting Victory Margin Update ---")

try:
    # Select the correct worksheet for victory margins
    victory_sheet = sh.worksheet('victory_margin_aging_cov')
    print("✓ Successfully connected to victory_margin_aging_cov sheet")
except Exception as e:
    print(f"❌ Error connecting to victory_margin_aging_cov sheet: {e}")
    print("Skipping Victory Margin Update")
    exit(0)

try:
    # Load the correct token IDs CSV
    victory_csv_path = "pt-2026/pt_margin_token.ids.csv"
    df_victory_tokens = pd.read_csv(victory_csv_path)
    print(f"Loaded {len(df_victory_tokens)} victory margin market definitions.")
    print("Loaded pt_margin_token.ids.csv data:")
    print(df_victory_tokens.head())
except Exception as e:
    print(f"❌ Error loading pt_margin_token.ids.csv: {e}")
    print("Skipping Victory Margin Update")
    exit(0)

try:
    # Find columns with "lower" and "upper" headers dynamically
    header_row = victory_sheet.get('1:1')[0]
    lower_col_idx_vic = None
    upper_col_idx_vic = None

    for i, header in enumerate(header_row):
        if header and 'lower' in str(header).lower():
            lower_col_idx_vic = i
            lower_col_letter_vic = num_to_col(i + 1)
            print(f"Found 'lower' column at index {i} (column {lower_col_letter_vic})")
        if header and 'upper' in str(header).lower():
            upper_col_idx_vic = i
            upper_col_letter_vic = num_to_col(i + 1)
            print(f"Found 'upper' column at index {i} (column {upper_col_letter_vic})")

    if lower_col_idx_vic is None or upper_col_idx_vic is None:
        print("❌ Error: Could not find 'lower' and/or 'upper' columns in victory_margin_aging_cov sheet header")
        print("Available headers:", [h for h in header_row if h])
        exit(0)

    # Read columns separately
    lower_values_vic = victory_sheet.get(f'{lower_col_letter_vic}2:{lower_col_letter_vic}50')
    upper_values_vic = victory_sheet.get(f'{upper_col_letter_vic}2:{upper_col_letter_vic}50')

    # Combine into pairs
    sheet_ranges_vic = []
    max_rows_vic = max(len(lower_values_vic), len(upper_values_vic))
    for i in range(max_rows_vic):
        lower_val = lower_values_vic[i][0] if i < len(lower_values_vic) and len(lower_values_vic[i]) > 0 else None
        upper_val = upper_values_vic[i][0] if i < len(upper_values_vic) and len(upper_values_vic[i]) > 0 else None
        sheet_ranges_vic.append([lower_val, upper_val])

    print(f"✓ Got victory margin ranges: {len(sheet_ranges_vic)} rows")

    # Find YES and NO header columns dynamically
    # Look for columns that might contain candidate names
    # Try to find a pattern - usually YES columns come before NO columns
    # We'll search for columns that contain candidate names from our data
    candidate_names = df_victory_tokens['name'].unique()
    print(f"Candidates in data: {candidate_names}")

    # Get a wider range of headers to find YES/NO columns
    # Assuming YES columns are before NO columns, search from column U onwards
    wide_headers = victory_sheet.get('U1:AZ1')[0]  # Search from U to AZ
    
    # Find YES columns (first occurrence of candidate names)
    yes_start_col_idx = None
    yes_headers_vic = []
    for i, header in enumerate(wide_headers):
        if header and any(name in str(header) for name in candidate_names):
            if yes_start_col_idx is None:
                yes_start_col_idx = i + 21  # U is column 21 (1-based)
            yes_headers_vic.append(header)
            if len(yes_headers_vic) >= 5:  # Assuming max 5 candidates
                break

    # Find NO columns (after YES columns, skip any separator columns)
    no_start_col_idx = yes_start_col_idx + len(yes_headers_vic) + 1 if yes_start_col_idx else None
    no_headers_vic = []
    if no_start_col_idx:
        for i in range(no_start_col_idx - 21, min(no_start_col_idx - 21 + 10, len(wide_headers))):
            if i < len(wide_headers) and wide_headers[i]:
                header = wide_headers[i]
                if any(name in str(header) for name in candidate_names) or 'NO' in str(header).upper():
                    no_headers_vic.append(header)
                    if len(no_headers_vic) >= 5:
                        break

    if not yes_headers_vic:
        print("❌ Error: Could not find YES header columns in victory_margin_aging_cov sheet")
        print("Available headers from U onwards:", [h for h in wide_headers if h])
        exit(0)

    # Create mappings
    yes_name_to_col_vic = {}
    for i, header in enumerate(yes_headers_vic):
        if header:
            col_letter = num_to_col(yes_start_col_idx + i)
            # Try to match candidate name from header
            for name in candidate_names:
                if name in str(header):
                    yes_name_to_col_vic[name] = col_letter
                    break

    no_name_to_col_vic = {}
    if no_start_col_idx and no_headers_vic:
        for i, header in enumerate(no_headers_vic):
            if header:
                col_letter = num_to_col(no_start_col_idx + i)
                # Try to match candidate name from header
                for name in candidate_names:
                    if name in str(header):
                        no_name_to_col_vic[name] = col_letter
                        break

    print(f"Victory YES column mapping: {yes_name_to_col_vic}")
    print(f"Victory NO column mapping: {no_name_to_col_vic}")

except Exception as e:
    print(f"❌ Error getting sheet data: {e}")
    print("Skipping Victory Margin Update")
    exit(0)

# Prepare batch update requests
yes_price_updates_vic = []
no_price_updates_vic = []

print("\nFetching prices for markets in pt_margin_token.ids.csv...")

for index, row in df_victory_tokens.iterrows():
    name = row['name']
    lo = row['lo']
    hi = row['hi']
    yes_token_id = row['yes_token_id']
    no_token_id = row['no_token_id']

    # Find the corresponding row in the Google Sheet
    target_row_index = -1
    for i, sheet_row in enumerate(sheet_ranges_vic):
        try:
            if len(sheet_row) >= 2:
                sheet_lo = int(float(sheet_row[0])) if sheet_row[0] else None
                sheet_hi = int(float(sheet_row[1])) if sheet_row[1] else None
                if sheet_lo == lo and sheet_hi == hi:
                    target_row_index = i + 2  # 1-based index, and data starts from row 2
                    break
        except (ValueError, IndexError, TypeError):
            continue

    if target_row_index == -1:
        print(f"Warning: No matching row found for {name} with victory range {lo}-{hi}")
        continue

    # Fetch YES price
    yes_price = ''
    try:
        orderbook = client.get_order_book(yes_token_id)
        _, yes_ask = get_extremes(orderbook)
        if yes_ask < 1:
            yes_price = yes_ask
    except Exception as e:
        print(f"Could not fetch YES price for {name} ({lo}-{hi}): {e}")

    # Fetch NO price (derived from YES token: 1 - max(bid))
    no_price = ''
    try:
        orderbook = client.get_order_book(yes_token_id)
        yes_bid, _ = get_extremes(orderbook)
        if yes_bid > 0:
            no_price = 1 - yes_bid
    except Exception as e:
        print(f"Could not fetch NO price for {name} ({lo}-{hi}): {e}")

    # Find column and prepare update
    if name in yes_name_to_col_vic and yes_price:
        col = yes_name_to_col_vic[name]
        cell = f"{col}{target_row_index}"
        yes_price_updates_vic.append({'range': cell, 'values': [[yes_price]]})
        print(f"  Prepared Victory YES update for {name} ({lo}-{hi}) at {cell}: {yes_price:.4f}")

    # For NO prices, check if name is in mapping, or use offset
    if no_price:
        if name in no_name_to_col_vic:
            col = no_name_to_col_vic[name]
            cell = f"{col}{target_row_index}"
            no_price_updates_vic.append({'range': cell, 'values': [[no_price]]})
            print(f"  Prepared Victory NO update for {name} ({lo}-{hi}) at {cell}: {no_price:.4f}")
        elif name in yes_name_to_col_vic and yes_start_col_idx:
            # Calculate NO column using offset
            yes_col_idx = list(yes_name_to_col_vic.keys()).index(name)
            no_col_letter = num_to_col(no_start_col_idx + yes_col_idx if no_start_col_idx else yes_start_col_idx + len(yes_headers_vic) + yes_col_idx)
            cell = f"{no_col_letter}{target_row_index}"
            no_price_updates_vic.append({'range': cell, 'values': [[no_price]]})
            print(f"  Prepared Victory NO update for {name} ({lo}-{hi}) at {cell}: {no_price:.4f} (using offset)")

    time.sleep(0.5)  # Be respectful to the API

# Execute batch updates
print("\nExecuting batch updates for victory tokens to Google Sheets...")

try:
    if yes_price_updates_vic:
        victory_sheet.batch_update(yes_price_updates_vic)
        print(f"✓ Successfully updated {len(yes_price_updates_vic)} Victory YES prices.")
    else:
        print("No Victory YES prices to update.")

    if no_price_updates_vic:
        victory_sheet.batch_update(no_price_updates_vic)
        print(f"✓ Successfully updated {len(no_price_updates_vic)} Victory NO prices.")
    else:
        print("No Victory NO prices to update.")

    print("\n🎉 Victory margin token functionality executed successfully!")

except Exception as e:
    print(f"❌ Error during victory token batch update: {e}")


