"""Reading multiple markets."""

from py_clob_client.client import ClobClient
import gspread
import pandas as pd
import os
from dotenv import load_dotenv
import time

load_dotenv()

# Connect to Polymarket
host: str = "https://clob.polymarket.com"
key: str = os.getenv('POLYMARKET_PRIVATE_KEY')
chain_id: int = 137
POLYMARKET_PROXY_ADDRESS: str = os.getenv('POLYMARKET_PROXY_ADDRESS')
sheetkey = "1es2J0O_Ig7RfnVHG3bHmX8SBjlMvrPwn4s1imYkxbwg"

# load data from GSheet
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)

# Initialize client
client = ClobClient(host, key=key, chain_id=chain_id, signature_type=2, funder=POLYMARKET_PROXY_ADDRESS)
creds = client.derive_api_key()
client.set_api_creds(creds)

# Different markets
markets = [
  {
    "name": "Motoristé",
    "sheet": "pravděpodobnosti_aktuální_aging_cov",
    "first_cell_yes": "AB131",
    "first_cell_no": "AB143",
    "opportunities": [
      {
        "token_id": "17214582629547852927716760684512138082254786333979249720214047614120364116640",
        "label": "3-5",
      },
      {
        "token_id": "64104997506099880061918851752583666958945335823947132440135386658993573929787",
        "label": "5-7",
      },
      {
        "token_id": "99721736934582524083350480936819448228417349127223814009482923979289475099211",
        "label": "7-9",
      },
      {
        "token_id": "13985124698614706139258217237538113591745137498709983147775661504681303148794",
        "label": ">9",
      },
      {
        "token_id": "5339302639488186206423482663378985926379574222367204383163355334647101043020",
        "label": "<3",
      },
    ]
  },
  {
    "name": "ANO",
    "sheet": "pravděpodobnosti_aktuální_aging_cov",
    "first_cell_yes": "V131",
    "first_cell_no": "V143",
    "opportunities": [
      {
        "token_id": "44893441083851309543247599835897759553567448471307820476136084578988482128269",
        "label": "<27",
      },
      {
        "token_id": "64104997506099880061918851752583666958945335823947132440135386658993573929787",
        "label": "27-30",
      },
      {
        "token_id": "93997883955615571772956694306764881840739936803791869723029394921523546826113",
        "label": "30-33",
      },
      {
        "token_id": "81623208862165926807694972513707921867138597391069874132181880555394852763571",
        "label": "33-36",
      },
      {
        "token_id": "114563309536528749390488662475257394441921087931650562623361207584160724689794",
        "label": ">36",
      },
    ]
  },
  {
    "name": "Přísaha",
    "sheet": "pravděpodobnosti_aktuální_aging_cov",
    "first_cell_yes": "AC131",
    "first_cell_no": "AC143",
    "opportunities": [
      {
        "token_id": "114715839910304061869573169086462305887241502021064197825758119448304812699266",
        "label": "<3",
      },
      {
        "token_id": "51296226825359979420855882022696521744571085921889355938987786293911759589314",
        "label": "3-5",
      },
      {
        "token_id": "39924063435457397022014354482002960520781973553409598590956706085647717197020",
        "label": "5-7",
      },
      {
        "token_id": "103530499037933994936147931738526257017167390667378234454809341335495531704748",
        "label": "7-9",
      },
      {
        "token_id": "7301896730697595746933275113920080017786648354277754961241724145066315837869",
        "label": ">9",
      },
    ]
  },
  {
    "name": "Stačilo!",
    "sheet": "pravděpodobnosti_aktuální_aging_cov",
    "first_cell_yes": "AA131",
    "first_cell_no": "AA143",
    "opportunities": [
      {
        "token_id": "42371473909219837771569014442357108002574564749341504257846807310667536186826",
        "label": "<5",
      },
      {
        "token_id": "73637027097955592724625045899275971395538315746483441149612494368544911365510",
        "label": "5-8",
      },
      {
        "token_id": "75258468834098949873695954641434021384612519257458786031597295435944430322460",
        "label": "8-11",
      },
      {
        "token_id": "44821988216199660607058330130777609291580206652618280797929348759036864574889",
        "label": "11-14",
      },
      {
        "token_id": "93442315160841775547086476191023710211583641888628165963143964099734414868090",
        "label": ">14",
      },
    ]
  },
  {
    "name": "SPOLU",
    "sheet": "pravděpodobnosti_aktuální_aging_cov",
    "first_cell_yes": "W131",
    "first_cell_no": "W143",
    "opportunities": [
      {
        "token_id": "12578390775922436388108189929377176721364218439347460687051764264703203724980",
        "label": "<17",
      },
      {
        "token_id": "45671930074789828006889199889240582814030273573003780648105047733556015231720",
        "label": "17-20",
      },
      {
        "token_id": "45671930074789828006889199889240582814030273573003780648105047733556015231720",
        "label": "20-23",
      },
      {
        "token_id": "54016316151510810387496895731352587242191837906725181057963094799753459350180",
        "label": "23-26",
      },
      {
        "token_id": "65398042001763755510146637952023013841475403168322035244352033717938967682819",
        "label": ">26",
      },
    ]
  },
  {
    "name": "STAN",
    "sheet": "pravděpodobnosti_aktuální_aging_cov",
    "first_cell_yes": "X131",
    "first_cell_no": "X143",
    "opportunities": [
      {
        "token_id": "111943627264740261356789062811417174889179791399413802499979950804119546249142",
        "label": "<7",
      },
      {
        "token_id": "17305267009158014651336828772171549044062040805354925006798487187449124742546",
        "label": "7-10",
      },
      {
        "token_id": "98351714581685995469720991424556153948873844923747603177473938189635867399138",
        "label": "10-13",
      },
      {
        "token_id": "83899854371299482258238408832713707952043404745268548825282394548340722158963",
        "label": "13-16",
      },
      {
        "token_id": "43260191024908787026484413013612695389010930425256531079375736593485185491859",
        "label": ">16",
      },
    ]
  },
  {
    "name": "SPD",
    "sheet": "pravděpodobnosti_aktuální_aging_cov",
    "first_cell_yes": "Y131",
    "first_cell_no": "Y143",
    "opportunities": [
      {
        "token_id": "20123190064354353070813080184323403881092090729921037653466403833843022591002",
        "label": "<7",
      },
      {
        "token_id": "73741254616265085688615588719190286280921993574678704138531863014023644829359",
        "label": "7-10",
      },
      {
        "token_id": "83884009387766945136437695229928047916936074676655039917335180994783267010019",
        "label": "10-13",
      },
      {
        "token_id": "96600411530960964556986901334364000651070803065689592124063971936888757582105",
        "label": "13-16",
      },
      {
        "token_id": "84233572326108703196453817703458095835413026412006409518873798788348884780517",
        "label": ">16",
      },
    ]
  },
  {
    "name": "Piráti",
    "sheet": "pravděpodobnosti_aktuální_aging_cov",
    "first_cell_yes": "Z131",
    "first_cell_no": "Z143",
    "opportunities": [
      {
        "token_id": "92089723743128084434183282586999127078883887330437135736698988959942372634019",
        "label": "<5",
      },
      {
        "token_id": "97493914161689837557625827813375623966790871243692488752813390649299653641307",
        "label": "5-8",
      },
      {
        "token_id": "52753005989061037158599838628666128448891375865797570704247461352708829304987",
        "label": "8-11",
      },
      {
        "token_id": "22428886288758375965364000799833405566455414318188408106685406043659532112851",
        "label": "11-14",
      },
      {
        "token_id": "38559811251086140643385439172567368355991260510664619500325443102872435767132",
        "label": ">14",
      },
    ]
  }
]

# get the current buy (ask) and buy no (bid) price for each token
for market in markets:
  sheet = sh.worksheet(market["sheet"])
  
  asks_to_write = []
  bids_to_write = []

  for opportunity in market["opportunities"]:
    token_id = opportunity["token_id"]
    orderbook = client.get_order_book(token_id)
    asks = [float(ask.price) for ask in orderbook.asks]
    ask = min(asks)
    bids = [float(bid.price) for bid in orderbook.bids]
    bid = 1 - max(bids)
    opportunity["ask"] = ask
    opportunity["bid"] = bid
    asks_to_write.extend([[opportunity['label']], [ask]])
    bids_to_write.extend([[opportunity['label']], [bid]])

  # and write them to GSheet
  sheet.update(asks_to_write, market["first_cell_yes"])
  sheet.update(bids_to_write, market["first_cell_no"])

  print("Done with market: " + market["name"] + ", waiting a bit...")
  if not market == markets[-1]:
    time.sleep(5)

# print the ask prices
for market in markets:
  print("\nMarket: " + market["name"])
  for opportunity in market["opportunities"]:
    print(f"{opportunity['label']}: {opportunity['ask']} {opportunity['bid']}")
