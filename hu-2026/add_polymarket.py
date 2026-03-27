"""Reading Polymarket odds for Hungarian parliamentary election winner."""

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
csv_path = "hu-2026/hungary-parliamentary-election-winner.csv"
df_tokens = pd.read_csv(csv_path)

print("Loaded token data:")
print(df_tokens.head())

# Filter to only Fidesz-KDNP and TISZA
parties_of_interest = ['Fidesz-KDNP', 'TISZA']
df_tokens = df_tokens[df_tokens['name'].isin(parties_of_interest)]

print(f"\nFiltered to parties of interest: {parties_of_interest}")
print(df_tokens)

# Google Sheet configuration
sheetkey = "1a3i0HfphGxlz-6_E04wYq30tG-gGvWQcyUc4-pNwI9U"
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)
sheet = sh.worksheet("seats_rank")

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

# Collect orderbook data for rank 1 (winning most seats)
print("\nFetching orderbook data for rank 1 (most seats)...")
polymarket_data = {}

for _, row in df_tokens.iterrows():
    party = row['name']
    yes_token_id = row['yes_token_id']
    
    print(f"Fetching data for {party}...")
    
    # Get YES token data (probability of winning most seats)
    try:
        yes_orderbook = client.get_order_book(yes_token_id)
        yes_bid, yes_ask = get_extremes(yes_orderbook)
        
        # Get shares available at different price levels
        shares_at_price = get_shares_at_price(yes_orderbook, yes_ask)
        shares_at_price_plus_1c = get_shares_at_price(yes_orderbook, yes_ask + 0.01)
        shares_at_price_plus_2c = get_shares_at_price(yes_orderbook, yes_ask + 0.02)
        
        # Store the data
        polymarket_data[party] = {
            'price': yes_ask,
            'shares_at_price': shares_at_price,
            'shares_at_price_plus_1c': shares_at_price_plus_1c,
            'shares_at_price_plus_2c': shares_at_price_plus_2c
        }
        
        print(f"  {party}: {yes_ask:.4f}")
        print(f"    Shares at price: {shares_at_price:.2f}")
        print(f"    Shares at price+1c: {shares_at_price_plus_1c:.2f}")
        print(f"    Shares at price+2c: {shares_at_price_plus_2c:.2f}")
        
    except Exception as e:
        print(f"Error fetching data for {party}: {e}")
        polymarket_data[party] = None
    
    # Wait a bit to be respectful to the API
    time.sleep(0.5)

print(f"\nCollected data for {len(polymarket_data)} parties")

# Prepare data for writing to Google Sheets
# We'll write starting at G1
# Format:
# G1: "Polymarket"  H1: "Shares@price"  I1: "Shares@price+1c"  J1: "Shares@price+2c"
# G2: <Fidesz price>  H2: <shares>  I2: <shares>  J2: <shares>
# G3: <Tisza price>   H3: <shares>  I3: <shares>  J3: <shares>
# G4: <MH price (empty)>
# G5: Timestamp

print("\nPreparing data for Google Sheets...")

# Map party names to match what's in the seats_rank sheet
# Fidesz-KDNP in CSV should match "Fidesz" in the sheet
# TISZA in CSV should match "Tisza" in the sheet
party_name_mapping = {
    'Fidesz-KDNP': 'Fidesz',
    'TISZA': 'Tisza'
}

# Get current timestamp
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Write to Google Sheets
print("\nWriting to Google Sheets...")

try:
    # First, read the existing party order from column A to match rows
    existing_parties = sheet.col_values(1)  # Column A
    
    # Find the row indices for each party (starting from row 2, after header)
    party_row_map = {}
    for i, party in enumerate(existing_parties):
        if party in ['Fidesz', 'Tisza', 'MH']:
            party_row_map[party] = i + 1  # 1-indexed
    
    print(f"Found party rows: {party_row_map}")
    
    # Write headers in row 1
    headers = [['Polymarket'], ['Shares@price'], ['Shares@price+1c'], ['Shares@price+2c']]
    sheet.update([['Polymarket']], range_name='G1')
    sheet.update([['Shares@price']], range_name='H1')
    sheet.update([['Shares@price+1c']], range_name='I1')
    sheet.update([['Shares@price+2c']], range_name='J1')
    print("✓ Headers written to G1:J1")
    
    # Write data for each party in columns G, H, I, J, matching the row positions
    for party_csv, party_sheet in party_name_mapping.items():
        if party_sheet in party_row_map and party_csv in polymarket_data and polymarket_data[party_csv] is not None:
            row = party_row_map[party_sheet]
            data = polymarket_data[party_csv]
            
            # Write price in column G
            sheet.update([[data['price']]], range_name=f'G{row}')
            
            # Write shares at price in column H
            sheet.update([[data['shares_at_price']]], range_name=f'H{row}')
            
            # Write shares at price+1c in column I
            sheet.update([[data['shares_at_price_plus_1c']]], range_name=f'I{row}')
            
            # Write shares at price+2c in column J
            sheet.update([[data['shares_at_price_plus_2c']]], range_name=f'J{row}')
            
            print(f"✓ {party_sheet} data written to row {row}:")
            print(f"    Price: {data['price']:.4f}, Shares: {data['shares_at_price']:.2f}, +1c: {data['shares_at_price_plus_1c']:.2f}, +2c: {data['shares_at_price_plus_2c']:.2f}")
    
    # Write empty values for MH if it exists
    if 'MH' in party_row_map:
        row = party_row_map['MH']
        sheet.update([['', '', '', '']], range_name=f'G{row}:J{row}')
        print(f"✓ MH (empty) written to row {row}")
    
    # Write timestamp below the last party
    timestamp_row = max(party_row_map.values()) + 1 if party_row_map else 5
    sheet.update([[current_time]], range_name=f'G{timestamp_row}')
    print(f"✓ Timestamp written to G{timestamp_row}: {current_time}")
    
    print("\n🎉 All data written successfully to Google Sheets!")
    
except Exception as e:
    print(f"❌ Error writing to Google Sheets: {e}")
    print("Please check your Google Sheets permissions and try again")
    import traceback
    traceback.print_exc()

print(f"\nSummary:")
print(f"- Processed {len(polymarket_data)} parties")
print(f"- Data written to seats_rank sheet in columns G-J")
print(f"- Parties: {list(party_name_mapping.values())}")
for party_csv, party_sheet in party_name_mapping.items():
    if party_csv in polymarket_data and polymarket_data[party_csv] is not None:
        data = polymarket_data[party_csv]
        print(f"  {party_sheet}: Price={data['price']:.4f}, Shares@price={data['shares_at_price']:.2f}, @+1c={data['shares_at_price_plus_1c']:.2f}, @+2c={data['shares_at_price_plus_2c']:.2f}")
