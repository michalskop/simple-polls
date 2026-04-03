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

# Process all parties with their rank numbers (no filtering)
print(f"\nProcessing all {len(df_tokens)} parties with rank numbers")
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

# Collect orderbook data for all parties with rank numbers
# Organize by rank: all YES for rank 1, all NO for rank 1, empty row, all YES for rank 2, all NO for rank 2, etc.
print("\nFetching orderbook data for all parties with rank numbers...")

# Group data by rank, separating YES and NO
rank_yes_data = {'1': [], '2': [], '3': []}
rank_no_data = {'1': [], '2': [], '3': []}

for _, row in df_tokens.iterrows():
    party_name = row['name']
    yes_token_id = row['yes_token_id']
    no_token_id = row['no_token_id']
    
    # Extract rank from party name (e.g., "TISZA 1" -> "1")
    rank = party_name.split()[-1]
    
    print(f"\nFetching data for {party_name}...")
    
    # Get YES token data
    try:
        yes_orderbook = client.get_order_book(yes_token_id)
        yes_bid, yes_ask = get_extremes(yes_orderbook)
        
        yes_shares_at_price = get_shares_at_price(yes_orderbook, yes_ask)
        yes_shares_at_price_plus_1c = get_shares_at_price(yes_orderbook, yes_ask + 0.01)
        yes_shares_at_price_plus_2c = get_shares_at_price(yes_orderbook, yes_ask + 0.02)
        
        print(f"  YES: {yes_ask:.4f}")
        print(f"    Shares at price: {yes_shares_at_price:.2f}")
        print(f"    Shares at price+1c: {yes_shares_at_price_plus_1c:.2f}")
        print(f"    Shares at price+2c: {yes_shares_at_price_plus_2c:.2f}")
        
        rank_yes_data[rank].append({
            'name': f"{party_name} YES",
            'price': yes_ask,
            'shares_at_price': yes_shares_at_price,
            'shares_at_price_plus_1c': yes_shares_at_price_plus_1c,
            'shares_at_price_plus_2c': yes_shares_at_price_plus_2c
        })
        
    except Exception as e:
        print(f"Error fetching YES data for {party_name}: {e}")
        rank_yes_data[rank].append({
            'name': f"{party_name} YES",
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
        
        no_shares_at_price = get_shares_at_price(no_orderbook, no_ask)
        no_shares_at_price_plus_1c = get_shares_at_price(no_orderbook, no_ask + 0.01)
        no_shares_at_price_plus_2c = get_shares_at_price(no_orderbook, no_ask + 0.02)
        
        print(f"  NO: {no_ask:.4f}")
        print(f"    Shares at price: {no_shares_at_price:.2f}")
        print(f"    Shares at price+1c: {no_shares_at_price_plus_1c:.2f}")
        print(f"    Shares at price+2c: {no_shares_at_price_plus_2c:.2f}")
        
        rank_no_data[rank].append({
            'name': f"{party_name} NO",
            'price': no_ask,
            'shares_at_price': no_shares_at_price,
            'shares_at_price_plus_1c': no_shares_at_price_plus_1c,
            'shares_at_price_plus_2c': no_shares_at_price_plus_2c
        })
        
    except Exception as e:
        print(f"Error fetching NO data for {party_name}: {e}")
        rank_no_data[rank].append({
            'name': f"{party_name} NO",
            'price': None,
            'shares_at_price': None,
            'shares_at_price_plus_1c': None,
            'shares_at_price_plus_2c': None
        })
    
    time.sleep(0.5)

# Combine all data: all YES for rank 1, all NO for rank 1, empty row, all YES for rank 2, all NO for rank 2, etc.
polymarket_data = []
empty_row = {
    'name': '',
    'price': '',
    'shares_at_price': '',
    'shares_at_price_plus_1c': '',
    'shares_at_price_plus_2c': ''
}

for rank in ['1', '2', '3']:
    # Add all YES tokens for this rank
    polymarket_data.extend(rank_yes_data[rank])
    # Add empty row between YES and NO
    polymarket_data.append(empty_row)
    # Add all NO tokens for this rank
    polymarket_data.extend(rank_no_data[rank])
    # Add empty row separator (except after the last group)
    if rank != '3':
        polymarket_data.append(empty_row)

print(f"\nCollected data for {len([d for d in polymarket_data if d['name']])} party-rank combinations (YES and NO)")

# Prepare data for writing to Google Sheets
# We'll write starting at G11
# Format:
# G11: "Party"  H11: "Polymarket"  I11: "Shares@price"  J11: "Shares@price+1c"  K11: "Shares@price+2c"
# G12+: data rows with party names and their probabilities

print("\nPreparing data for Google Sheets...")

# Get current timestamp
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Build the table
table = []

# Header row
table.append(['Party', 'Polymarket', 'Shares@price', 'Shares@price+1c', 'Shares@price+2c'])

# Data rows
for item in polymarket_data:
    table.append([
        item['name'],
        item['price'] if item['price'] is not None else '',
        item['shares_at_price'] if item['shares_at_price'] is not None else '',
        item['shares_at_price_plus_1c'] if item['shares_at_price_plus_1c'] is not None else '',
        item['shares_at_price_plus_2c'] if item['shares_at_price_plus_2c'] is not None else ''
    ])

# Write to Google Sheets
print("\nWriting to Google Sheets...")

try:
    # Calculate the range - starting at G11
    end_row = 10 + len(table)
    range_name = f'G11:K{end_row}'
    
    sheet.update(values=table, range_name=range_name)
    
    print(f"✓ All data written successfully to {range_name}")
    print(f"  Total rows: {len(table)} (including header)")
    
    # Write timestamp below the table
    timestamp_row = end_row + 1
    sheet.update([[current_time]], range_name=f'G{timestamp_row}')
    print(f"✓ Timestamp written to G{timestamp_row}: {current_time}")
    
    print("\n🎉 All data written successfully to Google Sheets!")
    
except Exception as e:
    print(f"❌ Error writing to Google Sheets: {e}")
    print("Please check your Google Sheets permissions and try again")
    import traceback
    traceback.print_exc()

print(f"\nSummary (seats_rank):")
print(f"- Processed {len([d for d in polymarket_data if d['name']])} party-rank combinations")
print(f"- Data written to seats_rank sheet in columns G-K starting at row 11")
print(f"- Parties with ranks:")
for item in polymarket_data:
    if item['name'] and item['price'] is not None and item['price'] != '':
        print(f"  {item['name']}: Price={item['price']:.4f}, Shares@price={item['shares_at_price']:.2f}, @+1c={item['shares_at_price_plus_1c']:.2f}, @+2c={item['shares_at_price_plus_2c']:.2f}")

# ============================================================
# POPULAR VOTE DATA
# ============================================================

print("\n" + "="*60)
print("PROCESSING POPULAR VOTE DATA")
print("="*60)

def process_popular_vote_csv(csv_path, file_label):
    """Process a popular vote CSV file and return data for YES and NO tokens."""
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

# Process both popular vote CSV files
print("\n" + "-"*60)
print("FIDESZ-KDNP POPULAR VOTE")
print("-"*60)
fidesz_yes, fidesz_no = process_popular_vote_csv(
    "hu-2026/hungary-election-fidesz-kdnp-of-popular-vote.csv",
    "hungary-election-fidesz-kdnp-of-popular-vote"
)

print("\n" + "-"*60)
print("TISZA POPULAR VOTE")
print("-"*60)
tisza_yes, tisza_no = process_popular_vote_csv(
    "hu-2026/hungary-election-tisza-of-popular-vote.csv",
    "hungary-election-tisza-of-popular-vote"
)

print("\n" + "-"*60)
print("PARTIES ENTERING PARLIAMENT")
print("-"*60)
parties_yes, parties_no = process_popular_vote_csv(
    "hu-2026/hungary-parliamentary-election-which-parties-enter-parliament.csv",
    "hungary-parliamentary-election-which-parties-enter-parliament"
)

# Combine all data with proper structure
empty_row = {
    'file': '',
    'name': '',
    'type': '',
    'price': '',
    'shares_at_price': '',
    'shares_at_price_plus_1c': '',
    'shares_at_price_plus_2c': ''
}

all_popular_vote_data = fidesz_yes + [empty_row] + fidesz_no + [empty_row] + tisza_yes + [empty_row] + tisza_no + [empty_row] + parties_yes + [empty_row] + parties_no

print(f"\nTotal popular vote data collected:")
print(f"  Fidesz YES: {len(fidesz_yes)} rows")
print(f"  Fidesz NO: {len(fidesz_no)} rows")
print(f"  TISZA YES: {len(tisza_yes)} rows")
print(f"  TISZA NO: {len(tisza_no)} rows")
print(f"  Parties entering parliament YES: {len(parties_yes)} rows")
print(f"  Parties entering parliament NO: {len(parties_no)} rows")
print(f"  Total (with empty rows): {len(all_popular_vote_data)} rows")

# Prepare data for writing to Google Sheets
print("\nPreparing popular vote data for Google Sheets...")

# Build the table
popular_vote_table = []

# Header row
popular_vote_table.append(['file', 'name', 'Polymarket', 'Shares@price', 'Shares@price+1c', 'Shares@price+2c'])

# Data rows
for item in all_popular_vote_data:
    popular_vote_table.append([
        item['file'],
        item['name'],
        item['price'] if item['price'] is not None else '',
        item['shares_at_price'] if item['shares_at_price'] is not None else '',
        item['shares_at_price_plus_1c'] if item['shares_at_price_plus_1c'] is not None else '',
        item['shares_at_price_plus_2c'] if item['shares_at_price_plus_2c'] is not None else ''
    ])

# Write to pravděpodobnosti_aktuální_aging_cov sheet
print("\nWriting popular vote data to Google Sheets...")

try:
    prob_sheet = sh.worksheet("pravděpodobnosti_aktuální_aging_cov")
    
    # Write timestamp at P2
    prob_sheet.update(values=[[current_time]], range_name='P2')
    print(f"✓ Timestamp written to P2: {current_time}")
    
    # Calculate the range - starting at Q19
    end_row = 18 + len(popular_vote_table)
    range_name = f'Q19:V{end_row}'
    
    prob_sheet.update(values=popular_vote_table, range_name=range_name)
    
    print(f"✓ Popular vote data written successfully to {range_name}")
    print(f"  Total rows: {len(popular_vote_table)} (including header)")
    
except Exception as e:
    print(f"❌ Error writing popular vote data to Google Sheets: {e}")
    print("Please check your Google Sheets permissions and try again")
    import traceback
    traceback.print_exc()

print(f"\n{'='*60}")
print("SUMMARY (popular vote)")
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
print(f"  - Empty row")
print(f"  - Parties entering parliament YES rows ({len(parties_yes)})")
print(f"  - Empty row")
print(f"  - Parties entering parliament NO rows ({len(parties_no)})")

# ============================================================
# SEAT DISTRIBUTION DATA
# ============================================================

print("\n" + "="*60)
print("PROCESSING SEAT DISTRIBUTION DATA")
print("="*60)

# List of CSV files to process for seat distribution
seat_csv_files = [
    ("hu-2026/hungary-election-fidesz-kdnp-wins-seats.csv", "hungary-election-fidesz-kdnp-wins-seats"),
    ("hu-2026/hungary-election-tisza-wins-at-least-seats.csv", "hungary-election-tisza-wins-at-least-seats"),
    ("hu-2026/hungary-election-tisza-wins-a-constitutional-majority.csv", "hungary-election-tisza-wins-a-constitutional-majority"),
    ("hu-2026/of-seats-won-by-fidesz-kdnp-in-hungary-parliamentary-election.csv", "of-seats-won-by-fidesz-kdnp-in-hungary-parliamentary-election"),
    ("hu-2026/of-seats-won-by-tisza-in-hungary-parliamentary-election.csv", "of-seats-won-by-tisza-in-hungary-parliamentary-election")
]

all_seat_data = []

for csv_path, file_label in seat_csv_files:
    print("\n" + "-"*60)
    print(f"Processing: {file_label}")
    print("-"*60)
    
    yes_data, no_data = process_popular_vote_csv(csv_path, file_label)
    
    # Add YES rows
    all_seat_data.extend(yes_data)
    # Add empty row
    all_seat_data.append(empty_row)
    # Add NO rows
    all_seat_data.extend(no_data)
    # Add empty row separator between files
    all_seat_data.append(empty_row)

print(f"\nTotal seat distribution data collected: {len(all_seat_data)} rows")

# Prepare data for writing to Google Sheets
print("\nPreparing seat distribution data for Google Sheets...")

# Build the table
seat_table = []

# Header row
seat_table.append(['file', 'name', 'Polymarket', 'Shares@price', 'Shares@price+1c', 'Shares@price+2c'])

# Helper function to sanitize float values
def sanitize_value(val):
    """Convert inf/nan to empty string, ensure all values are JSON-safe."""
    import math
    if val is None or val == '':
        return ''
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return ''
        # Convert float to string if it's very large to avoid JSON issues
        return float(val)
    if isinstance(val, int):
        # Very large integers should be converted to string
        if abs(val) > 1e15:
            return str(val)
        return val
    return str(val)

# Data rows
for item in all_seat_data:
    seat_table.append([
        sanitize_value(item['file']),
        sanitize_value(item['name']),
        sanitize_value(item['price']),
        sanitize_value(item['shares_at_price']),
        sanitize_value(item['shares_at_price_plus_1c']),
        sanitize_value(item['shares_at_price_plus_2c'])
    ])

# Write to seats_aging_cov sheet
print("\nWriting seat distribution data to Google Sheets...")

try:
    seats_cov_sheet = sh.worksheet("seats_aging_cov")
    
    # Calculate the range - starting at M40
    end_row = 39 + len(seat_table)
    range_name = f'M40:R{end_row}'
    
    # Debug: Check for problematic values
    import math
    for i, row in enumerate(seat_table):
        for j, val in enumerate(row):
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                print(f"WARNING: Found invalid float at row {i}, col {j}: {val}")
    
    seats_cov_sheet.update(values=seat_table, range_name=range_name)
    
    print(f"✓ Seat distribution data written successfully to {range_name}")
    print(f"  Total rows: {len(seat_table)} (including header)")
    
except Exception as e:
    print(f"❌ Error writing seat distribution data to Google Sheets: {e}")
    print("Please check your Google Sheets permissions and try again")
    import traceback
    traceback.print_exc()

print(f"\n{'='*60}")
print("SUMMARY (seat distribution)")
print(f"{'='*60}")
print(f"Data written to seats_aging_cov sheet in columns M-R starting at row 40")
print(f"Processed {len(seat_csv_files)} CSV files")

# ============================================================
# MARGIN OF VICTORY DATA
# ============================================================

print("\n" + "="*60)
print("PROCESSING MARGIN OF VICTORY DATA")
print("="*60)

# List of CSV files to process for margin of victory
margin_csv_files = [
    ("hu-2026/hungary-parliamentary-election-popular-vote-margin-of-victory.csv", "hungary-parliamentary-election-popular-vote-margin-of-victory"),
    ("hu-2026/hungary-election-popular-vote-margin-of-victory-higher-strikes-for-tisza.csv", "hungary-election-popular-vote-margin-of-victory-higher-strikes-for-tisza")
]

all_margin_data = []

for csv_path, file_label in margin_csv_files:
    print("\n" + "-"*60)
    print(f"Processing: {file_label}")
    print("-"*60)
    
    yes_data, no_data = process_popular_vote_csv(csv_path, file_label)
    
    # Add YES rows
    all_margin_data.extend(yes_data)
    # Add empty row
    all_margin_data.append(empty_row)
    # Add NO rows
    all_margin_data.extend(no_data)
    # Add empty row separator between files
    all_margin_data.append(empty_row)

print(f"\nTotal margin of victory data collected: {len(all_margin_data)} rows")

# Prepare data for writing to Google Sheets
print("\nPreparing margin of victory data for Google Sheets...")

# Build the table
margin_table = []

# Header row
margin_table.append(['file', 'name', 'Polymarket', 'Shares@price', 'Shares@price+1c', 'Shares@price+2c'])

# Data rows
for item in all_margin_data:
    margin_table.append([
        sanitize_value(item['file']),
        sanitize_value(item['name']),
        sanitize_value(item['price']),
        sanitize_value(item['shares_at_price']),
        sanitize_value(item['shares_at_price_plus_1c']),
        sanitize_value(item['shares_at_price_plus_2c'])
    ])

# Write to margin sheet
print("\nWriting margin of victory data to Google Sheets...")

try:
    margin_sheet = sh.worksheet("margin")
    
    # Calculate the range - starting at K21
    end_row = 20 + len(margin_table)
    range_name = f'K21:P{end_row}'
    
    # Debug: Check for problematic values
    import math
    for i, row in enumerate(margin_table):
        for j, val in enumerate(row):
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                print(f"WARNING: Found invalid float at row {i}, col {j}: {val}")
    
    margin_sheet.update(values=margin_table, range_name=range_name)
    
    print(f"✓ Margin of victory data written successfully to {range_name}")
    print(f"  Total rows: {len(margin_table)} (including header)")
    
except Exception as e:
    print(f"❌ Error writing margin of victory data to Google Sheets: {e}")
    print("Please check your Google Sheets permissions and try again")
    import traceback
    traceback.print_exc()

print(f"\n{'='*60}")
print("SUMMARY (margin of victory)")
print(f"{'='*60}")
print(f"Data written to margin sheet in columns K-P starting at row 21")
print(f"Processed {len(margin_csv_files)} CSV files")
