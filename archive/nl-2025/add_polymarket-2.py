"""Reading the markets for Netherlands 2025 elections."""

from py_clob_client.client import ClobClient
import gspread
import pandas as pd
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
csv_path = "nl-2025/nl_token_ids.csv"
df_tokens = pd.read_csv(csv_path)

print("Loaded token data:")
print(df_tokens.head())

# Google Sheet configuration
sheetkey = "1VbbRpS7lDBY-6fl7GsoHOq8lZjmn8L2nUgO9eEwlyW8"
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)
sheet = sh.worksheet("ranks")

# Get party order from Google Sheets (B1 to P1)
party_order = sheet.get('B1:P1')[0]  # Get first row, B to P
print(f"Party order from Google Sheets: {party_order}")

# Get all unique parties from the data
all_parties = df_tokens['party'].unique()
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
    party = row['party']
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
# Rows: ranks (1, 2, 3, 4)
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
    # Write YES data starting at B41
    yes_range = f"B22:{chr(ord('B') + len(party_order) - 1)}{22 + len(all_ranks) - 1}"
    print(f"Writing YES data to range: {yes_range}")
    sheet.update(yes_data, range_name=yes_range)
    print("✓ YES data written successfully")
    
    # Write NO data starting at B51
    no_range = f"B32:{chr(ord('B') + len(party_order) - 1)}{32 + len(all_ranks) - 1}"
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
print(f"- YES data written to rows 22-{22 + len(all_ranks) - 1}")
print(f"- NO data written to rows 32-{32 + len(all_ranks) - 1}")


# --- New Logic for Top5 and Intervals ---
print("\n--- Starting Top5 and Intervals Update ---")

try:
    # 1. Load new token data
    new_csv_path = "nl-2025/token_ids_2.csv"
    df_new_tokens = pd.read_csv(new_csv_path)
    print(f"Loaded {len(df_new_tokens)} new market definitions.")
except Exception as e:
    print(f"❌ Error loading token_ids_2.csv: {e}")
    print("Skipping Top5 and Intervals Update")
    exit(0)

# 2. Prepare for batch updates
top5_updates = []
intervals_updates = []
duels_updates = []

try:
    intervals_sheet = sh.worksheet("intervals")
    top5_sheet = sh.worksheet("top5")
    duels_sheet = sh.worksheet("duels")
    print("✓ Successfully connected to all sheets")
except Exception as e:
    print(f"❌ Error connecting to sheets: {e}")
    print("Skipping Top5 and Intervals Update")
    exit(0)

try:
    # Get party order from the top5 sheet header (B1 to Z1 for safety)
    party_order = top5_sheet.get('B1:Z1')[0]
    print(f"✓ Got party order from top5 sheet: {party_order}")

    # Get all data from sheets for matching
    all_intervals_data = intervals_sheet.get_all_values()
    intervals_header = all_intervals_data[0]
    intervals_data = all_intervals_data[1:]
    print(f"✓ Got intervals data: {len(intervals_data)} rows")

    duels_data = duels_sheet.get_all_records() # get_all_records is fine here if headers are unique
    print(f"✓ Got duels data: {len(duels_data)} rows")

    # Find column indexes from the header
    party_col_idx = intervals_header.index('party')
    type_col_idx = intervals_header.index('type')
    limits_col_idx = intervals_header.index('limits')
    print(f"✓ Found column indexes: party={party_col_idx}, type={type_col_idx}, limits={limits_col_idx}")
    
except Exception as e:
    print(f"❌ Error getting sheet data: {e}")
    print("Skipping Top5 and Intervals Update")
    exit(0)

# 3. Fetch data and prepare cell updates
for index, row in df_new_tokens.iterrows():
    market_type = row['type']
    party = row['party']
    limits = row['limits']
    yes_token_id = row['yes_token_id']
    
    print(f"Processing: {market_type} market for {party} ({limits})...")

    try:
        orderbook = client.get_order_book(yes_token_id)
        yes_bid, yes_ask = get_extremes(orderbook)
        time.sleep(0.5) # API rate limit

        # Calculate prices based on market type
        if market_type == 'duels':
            # For duels, YES price is p1, NO price is p2
            p1_price = yes_ask
            p2_price = 1 - yes_bid if yes_bid > 0 else 0
        else:
            # For other types, we use yes_ask and a calculated no_price
            no_price = 1 - yes_bid if yes_bid > 0 else 0

    except Exception as e:
        print(f"  -> Error fetching orderbook: {e}")
        continue

    if market_type == 'top5':
        found_in_top5 = False
        for col_idx, header_party in enumerate(party_order):
            if header_party == party:
                # +2 for B-based index
                sheet_col = col_idx + 2
                # YES price to row 11
                top5_updates.append(gspread.Cell(11, sheet_col, yes_ask))
                # NO price to row 12
                top5_updates.append(gspread.Cell(12, sheet_col, no_price))
                print(f"  -> Prepared top5 update for {party} at row 11/12, col {sheet_col}. YES: {yes_ask:.4f}, NO: {no_price:.4f}")
                found_in_top5 = True
        if not found_in_top5:
            print(f"  -> ERROR: Party '{party}' not found in top5 sheet header.")

    elif market_type in ['seats', 'percent']:
        found = False
        for i, sheet_row in enumerate(intervals_data):
            if (sheet_row[party_col_idx] == party and 
                sheet_row[type_col_idx] == market_type and 
                str(sheet_row[limits_col_idx]) == str(limits)):
                
                row_index = i + 2 # +2 for header and 1-based index
                intervals_updates.append(gspread.Cell(row_index, 7, yes_ask)) # Col G
                intervals_updates.append(gspread.Cell(row_index, 8, no_price)) # Col H
                print(f"  -> Prepared intervals update for {party} at row {row_index}. YES: {yes_ask:.4f}, NO: {no_price:.4f}")
                found = True
                break
        if not found:
            print(f"  -> ERROR: No matching row found in intervals sheet for {party}, {market_type}, {limits}.")

    elif market_type == 'duels':
        try:
            p1, p2 = party.split('|')
            found_duel = False
            for i, sheet_row in enumerate(duels_data):
                if (sheet_row['party 1'] == p1 and sheet_row['party 2'] == p2) or \
                   (sheet_row['party 1'] == p2 and sheet_row['party 2'] == p1):
                    
                    row_index = i + 2 # +2 for header and 1-based index
                    # The 'yes_token_id' corresponds to the first party in the duel
                    duels_updates.append(gspread.Cell(row_index, 9, p1_price)) # Col I
                    duels_updates.append(gspread.Cell(row_index, 10, p2_price)) # Col J
                    print(f"  -> Prepared duels update for {p1} vs {p2} at row {row_index}. P1: {p1_price:.4f}, P2: {p2_price:.4f}")
                    found_duel = True
                    break
            if not found_duel:
                print(f"  -> ERROR: No matching duel found for {p1} vs {p2}.")
        except Exception as e:
            print(f"  -> ERROR processing duel '{party}': {e}")

# 4. Execute batch updates
if top5_updates:
    print("\nWriting to top5 sheet...")
    top5_sheet.update_cells(top5_updates)
    print(f"✓ {len(top5_updates)} cells updated in 'top5' sheet.")

if intervals_updates:
    print("\nWriting to intervals sheet...")
    intervals_sheet.update_cells(intervals_updates)
    print(f"✓ {len(intervals_updates)} cells updated in 'intervals' sheet.")

if duels_updates:
    print("\nWriting to duels sheet...")
    duels_sheet.update_cells(duels_updates)
    print(f"✓ {len(duels_updates)} cells updated in 'duels' sheet.")

print("\n🎉 New market data processing complete!")
