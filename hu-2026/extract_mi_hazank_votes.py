"""
Extract historical Mi Hazánk vote counts for each district.
"""

import gspread
import json

sheetkey = "16N9OApgCo4nrDd4dlpIf0g7962F2mCXImwuhZakSTkg"

gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)

print("Extracting Mi Hazánk historical vote data...")

ws_mi_hazank = sh.worksheet('Mi Hazánk 22')
mi_hazank_data = ws_mi_hazank.get_all_values()

# Parse district Mi Hazánk votes
district_mi_hazank_votes = []

# Row 2 has the national percentage
national_pct_2022 = float(mi_hazank_data[1][1].replace(',', '.'))

for i in range(2, len(mi_hazank_data)):
    if mi_hazank_data[i][0] and mi_hazank_data[i][0] != 'Maďarsko':
        try:
            district_id = mi_hazank_data[i][0]
            votes_2022 = int(mi_hazank_data[i][1].replace('\xa0', '').replace(' ', ''))
            
            district_mi_hazank_votes.append({
                'id': district_id,
                'votes_2022': votes_2022
            })
        except (ValueError, IndexError):
            pass

print(f"Extracted Mi Hazánk votes for {len(district_mi_hazank_votes)} districts")
print(f"National Mi Hazánk 2022: {national_pct_2022}%")

# Save to JSON
with open('hu-2026/mi_hazank_district_votes.json', 'w', encoding='utf-8') as f:
    json.dump({
        'national_pct_2022': national_pct_2022,
        'districts': district_mi_hazank_votes
    }, f, indent=2, ensure_ascii=False)

print(f"✓ Saved to mi_hazank_district_votes.json")
print("\nSample districts:")
for d in district_mi_hazank_votes[:5]:
    print(f"  {d['id']}: {d['votes_2022']:,} votes")
