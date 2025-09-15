"""Reading the markets."""

from py_clob_client.client import ClobClient
import gspread
import pandas as pd

# Connect to Polymarket
host: str = "https://clob.polymarket.com"
key: str = os.getenv('POLYMARKET_PRIVATE_KEY')
chain_id: int = 137
POLYMARKET_PROXY_ADDRESS: str = os.getenv('POLYMARKET_PROXY_ADDRESS')

# Winner: https://polymarket.com/event/czech-republic-parliamentary-election-winner?tid=1757893795587
token_ids = [
  {
    "token_id": "21470855104479504653934185879444621466818607844880361830608605215149778386874",
    "label": "ANO"
  },
  {
    "token_id": "8766739263299548356587084519592809311201912770309199505351296368191696516398",
    "label": "Přísaha",
  },
  # {
  #   "token_id": "39761812608017675831934783654995036387944115678130409254838059375829456325383",
  #   "label": "ODS",
  # },
  {
    "token_id": "91048329092363318240306309820771569148110863491390098200148710658200427661410",
    "label": "Motoristé",
  },
  {
    "token_id": "34119374390638064095647488586524191449658987210460466210356606686364008054431",
    "label": "Stačilo!",
  },
  {
    "token_id": "39929012830266558669226622533000213228178487515507428822631344460328141441845",
    "label": "STAN",
  },
  {
    "token_id": "66439793274174764425365965414577330021367161915895469083983383596025692205407",
    "label": "Piráti",
  },
  {
    "token_id": "69945930169103687969692781161973947842850788216121585024728296900969325660026",
    "label": "SPD",
  }
]

# Read the current price for each token
# Initialize client
client = ClobClient(host, key=key, chain_id=chain_id, signature_type=2, funder=POLYMARKET_PROXY_ADDRESS)
creds = client.derive_api_key()
client.set_api_creds(creds)


def get_extremes(orderbook):
    bids = [float(bid.price) for bid in orderbook.bids]
    asks = [float(ask.price) for ask in orderbook.asks]
    return max(bids), min(asks)

# Get order book data
extremes = {}
for token_id in token_ids:
    orderbook = client.get_order_book(token_id["token_id"])
    bid, ask = get_extremes(orderbook)
    extremes[token_id["label"]] = {
        "bid": bid,
        "ask": ask,
    }

print(extremes)

# source sheet
sheetkey = "1es2J0O_Ig7RfnVHG3bHmX8SBjlMvrPwn4s1imYkxbwg"
path = "cz-2025/"

# load data from GSheet
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)

# pořadí_aktuální_aging_cov
sheet = sh.worksheet("pořadí_aktuální_aging_cov")

# parties -hardcoded
parties = ["ANO", "SPOLU", "STAN", "SPD", "Piráti", "Stačilo!", "Motoristé", "Přísaha"]

# create a list with the asks and bids
data = {
  'yes': [],
  'no': [],
}
for party in parties:
  if party in extremes:
    data['yes'].append(extremes[party]['ask'])
    data['no'].append(1 - extremes[party]['bid'])
  else:
    data['yes'].append('')
    data['no'].append('')

# create a dataframe
df = pd.DataFrame(data)

# write to GSheet, rotate the dataframe, start at B62
sheet.update(df.T.values.tolist(), range_name='B62')

