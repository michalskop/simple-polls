"""Reading the markets for Bucuresti mayor 2025 elections."""

from py_clob_client.client import ClobClient
import gspread
import pandas as pd
import os
import time
from datetime import datetime

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
csv_path = "ro-mayor-bucuresti-2025/token_ids.csv"
df_tokens = pd.read_csv(csv_path)

print("Loaded token data:")
print(df_tokens.head())

# Google Sheet configuration
sheetkey = "1tqV-b3IOpxIO5yUA1-8P0v94e3ggIMj-TcvfXYzVedM"
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)
sheet = sh.worksheet("pořadí_aktuální_aging_cov")

# Get party order from Google Sheets (B1 to P1)
party_order = sheet.get('B1:F1')[0]  # Get first row, B to P
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
    # Write current timestamp to B38
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Writing timestamp to B18: {current_time}")
    sheet.update([[current_time]], range_name='B18')
    print("✓ Timestamp written successfully")
    
    # Write YES data starting at B21
    yes_range = f"B21:{chr(ord('B') + len(party_order) - 1)}{21 + len(all_ranks) - 1}"
    print(f"Writing YES data to range: {yes_range}")
    sheet.update(yes_data, range_name=yes_range)
    print("✓ YES data written successfully")
    
    # Write NO data starting at B31
    no_range = f"B31:{chr(ord('B') + len(party_order) - 1)}{31 + len(all_ranks) - 1}"
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
print(f"- YES data written to rows 41-{41 + len(all_ranks) - 1}")
print(f"- NO data written to rows 51-{51 + len(all_ranks) - 1}")
