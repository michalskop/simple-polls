"""Reading Polymarket odds for Fidesz-KDNP and TISZA popular vote percentages."""

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

def process_csv_file(csv_path, file_label):
    """Process a CSV file and return data for YES and NO tokens."""
    df_tokens = pd.read_csv(csv_path)
    print(f"\nLoaded {file_label}:")
    print(df_tokens)
    
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
                'file': file_label,
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
                'file': file_label,
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
                'file': file_label,
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
                'file': file_label,
                'name': name,
                'type': 'no',
                'price': None,
                'shares_at_price': None,
                'shares_at_price_plus_1c': None,
                'shares_at_price_plus_2c': None
            })
        
        time.sleep(0.5)
    
    return yes_data, no_data

# Process both CSV files
print("="*60)
print("PROCESSING FIDESZ-KDNP POPULAR VOTE")
print("="*60)
fidesz_yes, fidesz_no = process_csv_file(
    "hu-2026/hungary-election-fidesz-kdnp-of-popular-vote.csv",
    "hungary-election-fidesz-kdnp-of-popular-vote"
)

print("\n" + "="*60)
print("PROCESSING TISZA POPULAR VOTE")
print("="*60)
tisza_yes, tisza_no = process_csv_file(
    "hu-2026/hungary-election-tisza-of-popular-vote.csv",
    "hungary-election-tisza-of-popular-vote"
)

# Combine all data with proper structure:
# Fidesz YES rows
# Empty row
# Fidesz NO rows
# Empty row
# TISZA YES rows
# Empty row
# TISZA NO rows

empty_row = {
    'file': '',
    'name': '',
    'type': '',
    'price': '',
    'shares_at_price': '',
    'shares_at_price_plus_1c': '',
    'shares_at_price_plus_2c': ''
}

all_data = fidesz_yes + [empty_row] + fidesz_no + [empty_row] + tisza_yes + [empty_row] + tisza_no

print(f"\nTotal data collected:")
print(f"  Fidesz YES: {len(fidesz_yes)} rows")
print(f"  Fidesz NO: {len(fidesz_no)} rows")
print(f"  TISZA YES: {len(tisza_yes)} rows")
print(f"  TISZA NO: {len(tisza_no)} rows")
print(f"  Total (with empty rows): {len(all_data)} rows")

# Prepare data for writing to Google Sheets
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

# Write to Google Sheets
print("\nWriting to Google Sheets...")

try:
    # Write timestamp at P2
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.update(values=[[current_time]], range_name='P2')
    print(f"✓ Timestamp written to P2: {current_time}")
    
    # Calculate the range - starting at Q9
    end_row = 8 + len(table)
    range_name = f'Q9:V{end_row}'
    
    sheet.update(values=table, range_name=range_name)
    
    print(f"✓ All data written successfully to {range_name}")
    print(f"  Total rows: {len(table)} (including header)")
    
except Exception as e:
    print(f"❌ Error writing to Google Sheets: {e}")
    print("Please check your Google Sheets permissions and try again")
    import traceback
    traceback.print_exc()

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Data written to pravděpodobnosti_aktuální_aging_cov sheet in columns Q-V")
print(f"Structure:")
print(f"  - Fidesz YES rows ({len(fidesz_yes)})")
print(f"  - Empty row")
print(f"  - Fidesz NO rows ({len(fidesz_no)})")
print(f"  - Empty row")
print(f"  - TISZA YES rows ({len(tisza_yes)})")
print(f"  - Empty row")
print(f"  - TISZA NO rows ({len(tisza_no)})")
