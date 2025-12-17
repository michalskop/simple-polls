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
csv_path = "pt-2026-2/pt_token_ids.csv"
df_tokens = pd.read_csv(csv_path)

print("Loaded token data:")
print(df_tokens.head())

# Google Sheet configuration
sheetkey = "1m0_GDjalz9ukDfiDlPwzTnBamQFIXtlWgHbGFp9I_7Q"
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)
sheet = sh.worksheet("poradi")

# Get all unique parties from the data
all_parties = df_tokens['name'].unique()
party_order = [party for party in all_parties if party]
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
# Rows: ranks (1)
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
    # Write current timestamp to H8
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Writing timestamp to H8: {current_time}")
    sheet.update([[current_time]], range_name='H8')
    print("✓ Timestamp written successfully")
    
    # Write YES data starting at H10
    yes_range = f"H10:{chr(ord('H') + len(party_order) - 1)}{10 + len(all_ranks) - 1}"
    print(f"Writing YES data to range: {yes_range}")
    sheet.update(yes_data, range_name=yes_range)
    print("✓ YES data written successfully")
    
    # Write NO data starting at H14
    no_range = f"H14:{chr(ord('H') + len(party_order) - 1)}{14 + len(all_ranks) - 1}"
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
print(f"- YES data written to rows H10-H{10 + len(all_ranks) - 1}")
print(f"- NO data written to rows H14-H{14 + len(all_ranks) - 1}")

# --- Functionality starts here: pravděpodobnosti_aktuální_aging_cov ---

# print("\nStarting new functionality: Writing prices based on pt_token_ids.csv")

# sheet = sh.worksheet("pravděpodobnosti_aktuální_aging_cov")

# # Load the correct token IDs CSV
# pt_csv_path = "pt-2026/pt_token_ids.csv"
# df_pt_tokens = pd.read_csv(pt_csv_path)

# print("Loaded pt_token_ids.csv data:")
# print(df_pt_tokens.head())

# # Get headers from the sheet for name matching (AB1:AL1)
# headers = sheet.get('AB1:AL1')[0]
# yes_headers = headers[:5]  # AB to AF
# no_headers = headers[6:]   # AH to AL

# # Create a mapping from name to column index
# yes_name_to_col = {name: 'A' + chr(ord('B') + i) for i, name in enumerate(yes_headers)}
# no_name_to_col = {name: 'A' + chr(ord('H') + i) for i, name in enumerate(no_headers)}

# print(f"YES column mapping: {yes_name_to_col}")
# print(f"NO column mapping: {no_name_to_col}")

# # Get lo/hi ranges from the sheet (columns H and I)
# sheet_ranges = sheet.get('H2:I50') # Assuming max 50 rows

# # Prepare batch update requests
# yes_price_updates = []
# no_price_updates = []

# print("\nFetching prices for markets in pt_token_ids.csv...")

# for index, row in df_pt_tokens.iterrows():
#     name = row['name']
#     lo = row['lo']
#     hi = row['hi']
#     yes_token_id = row['yes_token_id']
#     no_token_id = row['no_token_id']

#     # Find the corresponding row in the Google Sheet
#     target_row_index = -1
#     for i, sheet_row in enumerate(sheet_ranges):
#         try:
#             sheet_lo = int(sheet_row[0])
#             sheet_hi = int(sheet_row[1])
#             if sheet_lo == lo and sheet_hi == hi:
#                 target_row_index = i + 2  # 1-based index, and data starts from row 2
#                 break
#         except (ValueError, IndexError):
#             continue

#     if target_row_index == -1:
#         print(f"Warning: No matching row found for {name} with range {lo}-{hi}")
#         continue

#     # Fetch YES price
#     yes_price = ''
#     try:
#         orderbook = client.get_order_book(yes_token_id)
#         _, yes_ask = get_extremes(orderbook)
#         if yes_ask < 1:
#             yes_price = f"{yes_ask:.4f}"
#     except Exception as e:
#         print(f"Could not fetch YES price for {name} ({lo}-{hi}): {e}")

#     # Fetch NO price
#     no_price = ''
#     try:
#         orderbook = client.get_order_book(no_token_id)
#         _, no_ask = get_extremes(orderbook)
#         if no_ask < 1:
#             no_price = f"{no_ask:.4f}"
#     except Exception as e:
#         print(f"Could not fetch NO price for {name} ({lo}-{hi}): {e}")

#     # Find column and prepare update
#     if name in yes_name_to_col:
#         col = yes_name_to_col[name]
#         cell = f"{col}{target_row_index}"
#         yes_price_updates.append({'range': cell, 'values': [[yes_price]]})
#         print(f"  Prepared YES update for {name} ({lo}-{hi}) at {cell}: {yes_price}")

#     if name in no_name_to_col:
#         col = no_name_to_col[name]
#         cell = f"{col}{target_row_index}"
#         no_price_updates.append({'range': cell, 'values': [[no_price]]})
#         print(f"  Prepared NO update for {name} ({lo}-{hi}) at {cell}: {no_price}")

#     time.sleep(0.5) # Be respectful to the API

# # Execute batch updates
# print("\nExecuting batch updates to Google Sheets...")

# try:
#     if yes_price_updates:
#         sheet.batch_update(yes_price_updates)
#         print(f"✓ Successfully updated {len(yes_price_updates)} YES prices.")
#     else:
#         print("No YES prices to update.")

#     if no_price_updates:
#         sheet.batch_update(no_price_updates)
#         print(f"✓ Successfully updated {len(no_price_updates)} NO prices.")
#     else:
#         print("No NO prices to update.")

#     print("\n🎉 New functionality executed successfully!")

# except Exception as e:
#     print(f"❌ Error during batch update: {e}")


# --- New functionality for victory tokens ---

# print("\nStarting new functionality: Writing prices based on ro_victory_token_ids.csv")

# # Select the correct worksheet for victory margins
# victory_sheet = sh.worksheet('victory_margin_aging_cov')

# # Load the correct token IDs CSV
# victory_csv_path = "ro-mayor-bucuresti-2025/ro_victory_token_ids.csv"
# df_victory_tokens = pd.read_csv(victory_csv_path)

# print("Loaded ro_victory_token_ids.csv data:")
# print(df_victory_tokens.head())

# # Get headers from the sheet for name matching
# yes_headers_vic = victory_sheet.get('U1:Y1')[0]
# no_headers_vic = victory_sheet.get('AA1:AE1')[0]

# # Create a mapping from name to column index
# yes_name_to_col_vic = {name: chr(ord('U') + i) for i, name in enumerate(yes_headers_vic)}
# no_name_to_col_vic = {name: 'A' + chr(ord('A') + i) for i, name in enumerate(no_headers_vic)}

# print(f"Victory YES column mapping: {yes_name_to_col_vic}")
# print(f"Victory NO column mapping: {no_name_to_col_vic}")

# # Get lo/hi ranges from the sheet (columns A and B)
# sheet_ranges_vic = victory_sheet.get('A2:B50') # Assuming max 50 rows

# # Prepare batch update requests
# yes_price_updates_vic = []
# no_price_updates_vic = []

# print("\nFetching prices for markets in ro_victory_token_ids.csv...")

# for index, row in df_victory_tokens.iterrows():
#     name = row['name']
#     lo = row['lo']
#     hi = row['hi']
#     yes_token_id = row['yes_token_id']
#     no_token_id = row['no_token_id']

#     # Find the corresponding row in the Google Sheet
#     target_row_index = -1
#     for i, sheet_row in enumerate(sheet_ranges_vic):
#         try:
#             sheet_lo = int(sheet_row[0])
#             sheet_hi = int(sheet_row[1])
#             if sheet_lo == lo and sheet_hi == hi:
#                 target_row_index = i + 2  # 1-based index, and data starts from row 2
#                 break
#         except (ValueError, IndexError):
#             continue

#     if target_row_index == -1:
#         print(f"Warning: No matching row found for {name} with victory range {lo}-{hi}")
#         continue

#     # Fetch YES price
#     yes_price = ''
#     try:
#         orderbook = client.get_order_book(yes_token_id)
#         _, yes_ask = get_extremes(orderbook)
#         if yes_ask < 1:
#             yes_price = f"{yes_ask:.4f}"
#     except Exception as e:
#         print(f"Could not fetch YES price for {name} ({lo}-{hi}): {e}")

#     # Fetch NO price
#     no_price = ''
#     try:
#         orderbook = client.get_order_book(no_token_id)
#         _, no_ask = get_extremes(orderbook)
#         if no_ask < 1:
#             no_price = f"{no_ask:.4f}"
#     except Exception as e:
#         print(f"Could not fetch NO price for {name} ({lo}-{hi}): {e}")

#     # Find column and prepare update
#     if name in yes_name_to_col_vic:
#         col = yes_name_to_col_vic[name]
#         cell = f"{col}{target_row_index}"
#         yes_price_updates_vic.append({'range': cell, 'values': [[yes_price]]})
#         print(f"  Prepared Victory YES update for {name} ({lo}-{hi}) at {cell}: {yes_price}")

#     if name in no_name_to_col_vic:
#         col = no_name_to_col_vic[name]
#         cell = f"{col}{target_row_index}"
#         no_price_updates_vic.append({'range': cell, 'values': [[no_price]]})
#         print(f"  Prepared Victory NO update for {name} ({lo}-{hi}) at {cell}: {no_price}")

#     time.sleep(0.5) # Be respectful to the API

# # Execute batch updates
# print("\nExecuting batch updates for victory tokens to Google Sheets...")

# try:
#     if yes_price_updates_vic:
#         victory_sheet.batch_update(yes_price_updates_vic)
#         print(f"✓ Successfully updated {len(yes_price_updates_vic)} Victory YES prices.")
#     else:
#         print("No Victory YES prices to update.")

#     if no_price_updates_vic:
#         victory_sheet.batch_update(no_price_updates_vic)
#         print(f"✓ Successfully updated {len(no_price_updates_vic)} Victory NO prices.")
#     else:
#         print("No Victory NO prices to update.")

#     print("\n🎉 Victory token functionality executed successfully!")

# except Exception as e:
#     print(f"❌ Error during victory token batch update: {e}")


