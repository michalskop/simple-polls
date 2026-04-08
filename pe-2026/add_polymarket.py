"""Reading Polymarket odds for Peruvian presidential election first round winner."""

from py_clob_client.client import ClobClient
import gspread
import pandas as pd
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

host: str = "https://clob.polymarket.com"
key: str = os.getenv('POLYMARKET_PRIVATE_KEY')
chain_id: int = 137
POLYMARKET_PROXY_ADDRESS: str = os.getenv('POLYMARKET_PROXY_ADDRESS')

client = ClobClient(host, key=key, chain_id=chain_id, signature_type=2, funder=POLYMARKET_PROXY_ADDRESS)
creds = client.derive_api_key()
client.set_api_creds(creds)

csv_path = "pe-2026/peru-presidential-election-first-round-winner.csv"
df_tokens = pd.read_csv(csv_path)

print("Loaded token data:")
print(df_tokens.head())

sheetkey = "1vy06C-n7gHndbYV8yEl1BADRPjtolBEbLRG_7d0Cdkg"
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)
sheet = sh.worksheet("pořadí_aktuální_aging_cov")

def get_extremes(orderbook):
    """Get bid and ask prices from orderbook."""
    bids = [float(bid.price) for bid in orderbook.bids]
    asks = [float(ask.price) for ask in orderbook.asks]
    return max(bids, default=0), min(asks, default=1)

def get_shares_at_price(orderbook, target_price):
    """Get total shares available at a specific price level."""
    total_shares = 0
    for ask in orderbook.asks:
        if abs(float(ask.price) - target_price) < 0.0001:
            total_shares += float(ask.size)
    return total_shares

print("\nFetching orderbook data for first 4 places...")

rank_1_candidates = []
rank_2_candidates = []
rank_3_candidates = []
rank_4_candidates = []

for _, row in df_tokens.iterrows():
    name = row['name']
    
    parts = name.split()
    rank = parts[-1]
    candidate_name = ' '.join(parts[:-1])
    
    if rank == '1':
        rank_1_candidates.append((candidate_name, row['yes_token_id'], row['no_token_id']))
    elif rank == '2':
        rank_2_candidates.append((candidate_name, row['yes_token_id'], row['no_token_id']))
    elif rank == '3':
        rank_3_candidates.append((candidate_name, row['yes_token_id'], row['no_token_id']))
    elif rank == '4':
        rank_4_candidates.append((candidate_name, row['yes_token_id'], row['no_token_id']))

all_candidates = [rank_1_candidates, rank_2_candidates, rank_3_candidates, rank_4_candidates]

yes_prices_by_rank = [[], [], [], []]
no_prices_by_rank = [[], [], [], []]

for rank_idx, rank_candidates in enumerate(all_candidates):
    print(f"\nProcessing rank {rank_idx + 1} candidates...")
    
    for candidate_name, yes_token_id, no_token_id in rank_candidates:
        print(f"\nFetching data for {candidate_name} (rank {rank_idx + 1})...")
        
        try:
            yes_orderbook = client.get_order_book(yes_token_id)
            yes_bid, yes_ask = get_extremes(yes_orderbook)
            print(f"  YES: {yes_ask:.4f}")
            yes_prices_by_rank[rank_idx].append(yes_ask)
        except Exception as e:
            print(f"Error fetching YES data for {candidate_name}: {e}")
            yes_prices_by_rank[rank_idx].append(None)
        
        time.sleep(0.5)
        
        try:
            no_orderbook = client.get_order_book(no_token_id)
            no_bid, no_ask = get_extremes(no_orderbook)
            print(f"  NO: {no_ask:.4f}")
            no_prices_by_rank[rank_idx].append(no_ask)
        except Exception as e:
            print(f"Error fetching NO data for {candidate_name}: {e}")
            no_prices_by_rank[rank_idx].append(None)
        
        time.sleep(0.5)

print("\nWriting to Google Sheets...")

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

try:
    yes_data = [
        [price if price is not None else '' for price in yes_prices_by_rank[0]],
        [price if price is not None else '' for price in yes_prices_by_rank[1]],
        [price if price is not None else '' for price in yes_prices_by_rank[2]],
        [price if price is not None else '' for price in yes_prices_by_rank[3]]
    ]
    sheet.update(values=yes_data, range_name='O2:Y5')
    print(f"✓ YES prices written to O2:Y5")
    
    no_data = [
        [price if price is not None else '' for price in no_prices_by_rank[0]],
        [price if price is not None else '' for price in no_prices_by_rank[1]],
        [price if price is not None else '' for price in no_prices_by_rank[2]],
        [price if price is not None else '' for price in no_prices_by_rank[3]]
    ]
    sheet.update(values=no_data, range_name='O16:Y19')
    print(f"✓ NO prices written to O16:Y19")
    
    sheet.update(values=[[current_time]], range_name='N1')
    print(f"✓ Timestamp written to N1: {current_time}")
    
    print("\n🎉 All data written successfully to Google Sheets!")
    
except Exception as e:
    print(f"❌ Error writing to Google Sheets: {e}")
    print("Please check your Google Sheets permissions and try again")
    import traceback
    traceback.print_exc()

print(f"\nSummary (First Round Winner):")
total_yes = sum(len(row) for row in yes_prices_by_rank)
total_no = sum(len(row) for row in no_prices_by_rank)
print(f"- Processed {total_yes} YES prices and {total_no} NO prices")
print(f"- YES prices written to O2:Y5 (4 rows x 11 columns)")
print(f"- NO prices written to O16:Y19 (4 rows x 11 columns)")
print(f"- Timestamp written to N1")

# ============================================================
# RUNOFF DATA
# ============================================================

print("\n" + "="*60)
print("PROCESSING RUNOFF DATA")
print("="*60)

runoff_csv_path = "pe-2026/which-candidates-advance-to-2026-peru-presidential-runoff.csv"
df_runoff = pd.read_csv(runoff_csv_path)

print(f"\nLoaded runoff data:")
print(df_runoff)

top_2_sheet = sh.worksheet("top_2_cov")

runoff_yes_data = []
runoff_no_data = []

for _, row in df_runoff.iterrows():
    name = row['name']
    yes_token_id = row['yes_token_id']
    no_token_id = row['no_token_id']
    
    print(f"\nFetching runoff data for {name}...")
    
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
        
        runoff_yes_data.append([name, yes_ask, yes_shares_at_price, yes_shares_at_price_plus_1c, yes_shares_at_price_plus_2c])
        
    except Exception as e:
        print(f"Error fetching YES data for {name}: {e}")
        runoff_yes_data.append([name, '', '', '', ''])
    
    time.sleep(0.5)
    
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
        
        runoff_no_data.append([name, no_ask, no_shares_at_price, no_shares_at_price_plus_1c, no_shares_at_price_plus_2c])
        
    except Exception as e:
        print(f"Error fetching NO data for {name}: {e}")
        runoff_no_data.append([name, '', '', '', ''])
    
    time.sleep(0.5)

print("\nWriting runoff data to Google Sheets...")

try:
    yes_end_row = 20 + len(runoff_yes_data)
    top_2_sheet.update(values=runoff_yes_data, range_name=f'A21:E{yes_end_row}')
    print(f"✓ Runoff YES data written to A21:E{yes_end_row}")
    
    no_end_row = 35 + len(runoff_no_data)
    top_2_sheet.update(values=runoff_no_data, range_name=f'A36:E{no_end_row}')
    print(f"✓ Runoff NO data written to A36:E{no_end_row}")
    
    print("\n🎉 Runoff data written successfully to Google Sheets!")
    
except Exception as e:
    print(f"❌ Error writing runoff data to Google Sheets: {e}")
    print("Please check your Google Sheets permissions and try again")
    import traceback
    traceback.print_exc()

print(f"\nSummary (Runoff):")
print(f"- Processed {len(runoff_yes_data)} YES entries and {len(runoff_no_data)} NO entries")
print(f"- YES data written to A21:E{20 + len(runoff_yes_data)}")
print(f"- NO data written to A36:E{35 + len(runoff_no_data)}")
