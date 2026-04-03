"""Reading Polymarket odds for Fidesz-KDNP popular vote percentages."""

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
csv_path = "hu-2026/hungary-election-fidesz-kdnp-of-popular-vote.csv"
df_tokens = pd.read_csv(csv_path)

print("Loaded token data:")
print(df_tokens)

# Google Sheet configuration
sheetkey = "1a3i0HfphGxlz-6_E04wYq30tG-gGvWQcyUc4-pNwI9U"
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)
sheet = sh.worksheet("pravděpodobnosti_aktuální_aging_cov")

def get_extremes(orderbook):
    """Get bid and ask prices from orderbook."""
    bids = [float(bid.price) for bid in orderbook.bids]
    asks = [float(ask.price) for ask in orderbook.asks]
    return max(bids, default=0), min(asks, default=1)

def get_shares_at_price(orderbook, target_price):
    """Get total shares available at a specific price level."""
    total_shares = 0
    for ask in orderbook.asks:
        if abs(float(ask.price) - target_price) < 0.0001:  # Price match with small tolerance
            total_shares += float(ask.size)
    return total_shares

# Collect orderbook data
print("\nFetching orderbook data...")
yes_data = []
no_data = []

for _, row in df_tokens.iterrows():
    name = row['name']
    yes_token_id = row['yes_token_id']
    no_token_id = row['no_token_id']
    
    print(f"\nFetching data for {name}...")
    
    # Get YES token data
    try:
        yes_orderbook = client.get_order_book(yes_token_id)
        yes_bid, yes_ask = get_extremes(yes_orderbook)
        
        # Get shares available at different price levels
        yes_shares_at_price = get_shares_at_price(yes_orderbook, yes_ask)
        yes_shares_at_price_plus_1c = get_shares_at_price(yes_orderbook, yes_ask + 0.01)
        yes_shares_at_price_plus_2c = get_shares_at_price(yes_orderbook, yes_ask + 0.02)
        
        print(f"  YES: {yes_ask:.4f}")
        print(f"    Shares at price: {yes_shares_at_price:.2f}")
        print(f"    Shares at price+1c: {yes_shares_at_price_plus_1c:.2f}")
        print(f"    Shares at price+2c: {yes_shares_at_price_plus_2c:.2f}")
        
        # Store YES data
        yes_data.append({
            'file': 'hungary-election-fidesz-kdnp-of-popular-vote',
            'name': name,
            'type': 'yes',
            'price': yes_ask,
            'shares_at_price': yes_shares_at_price,
            'shares_at_price_plus_1c': yes_shares_at_price_plus_1c,
            'shares_at_price_plus_2c': yes_shares_at_price_plus_2c
        })
        
    except Exception as e:
        print(f"Error fetching YES data for {name}: {e}")
        yes_data.append({
            'file': 'hungary-election-fidesz-kdnp-of-popular-vote',
            'name': name,
            'type': 'yes',
            'price': None,
            'shares_at_price': None,
            'shares_at_price_plus_1c': None,
            'shares_at_price_plus_2c': None
        })
    
    time.sleep(0.5)
    
    # Get NO token data
    try:
        no_orderbook = client.get_order_book(no_token_id)
        no_bid, no_ask = get_extremes(no_orderbook)
        
        # Get shares available at different price levels
        no_shares_at_price = get_shares_at_price(no_orderbook, no_ask)
        no_shares_at_price_plus_1c = get_shares_at_price(no_orderbook, no_ask + 0.01)
        no_shares_at_price_plus_2c = get_shares_at_price(no_orderbook, no_ask + 0.02)
        
        print(f"  NO: {no_ask:.4f}")
        print(f"    Shares at price: {no_shares_at_price:.2f}")
        print(f"    Shares at price+1c: {no_shares_at_price_plus_1c:.2f}")
        print(f"    Shares at price+2c: {no_shares_at_price_plus_2c:.2f}")
        
        # Store NO data
        no_data.append({
            'file': 'hungary-election-fidesz-kdnp-of-popular-vote',
            'name': name,
            'type': 'no',
            'price': no_ask,
            'shares_at_price': no_shares_at_price,
            'shares_at_price_plus_1c': no_shares_at_price_plus_1c,
            'shares_at_price_plus_2c': no_shares_at_price_plus_2c
        })
        
    except Exception as e:
        print(f"Error fetching NO data for {name}: {e}")
        no_data.append({
            'file': 'hungary-election-fidesz-kdnp-of-popular-vote',
            'name': name,
            'type': 'no',
            'price': None,
            'shares_at_price': None,
            'shares_at_price_plus_1c': None,
            'shares_at_price_plus_2c': None
        })
    
    time.sleep(0.5)

# Combine data: all YES rows, then empty row, then all NO rows
all_data = yes_data + [{
    'file': 'hungary-election-fidesz-kdnp-of-popular-vote',
    'name': '',
    'type': '',
    'price': '',
    'shares_at_price': '',
    'shares_at_price_plus_1c': '',
    'shares_at_price_plus_2c': ''
}] + no_data

print(f"\nCollected data for {len(df_tokens)} conditions")

# Prepare data for writing to Google Sheets
# Starting at Q2
# Format:
# Q1: "file"  R1: "name"  S1: "Polymarket"  T1: "Shares@price"  U1: "Shares@price+1c"  V1: "Shares@price+2c"
# Q2+: data rows

print("\nPreparing data for Google Sheets...")

# Build the table
table = []

# Header row
table.append(['file', 'name', 'Polymarket', 'Shares@price', 'Shares@price+1c', 'Shares@price+2c'])

# Data rows
for item in all_data:
    table.append([
        item['file'],
        item['name'],
        item['price'] if item['price'] is not None else '',
        item['shares_at_price'] if item['shares_at_price'] is not None else '',
        item['shares_at_price_plus_1c'] if item['shares_at_price_plus_1c'] is not None else '',
        item['shares_at_price_plus_2c'] if item['shares_at_price_plus_2c'] is not None else ''
    ])

# Write to Google Sheets in one call
print("\nWriting to Google Sheets...")

try:
    # Calculate the range - starting at Q1
    end_row = len(table)
    range_name = f'Q1:V{end_row}'
    
    sheet.update(values=table, range_name=range_name)
    
    print(f"✓ All data written successfully to {range_name}")
    print(f"  Total rows: {len(table)} (including header)")
    
except Exception as e:
    print(f"❌ Error writing to Google Sheets: {e}")
    print("Please check your Google Sheets permissions and try again")
    import traceback
    traceback.print_exc()

print(f"\nSummary:")
print(f"- Processed {len(df_tokens)} conditions")
print(f"- Total data rows: {len(all_data)} (including empty rows)")
print(f"- Data written to pravděpodobnosti_aktuální_aging_cov sheet in columns Q-V")
