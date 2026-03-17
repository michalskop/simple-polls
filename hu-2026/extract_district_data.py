"""
Extract district data from Google Sheets and save to local JSON files.
This is a one-time script to create the cached data files.
"""

import gspread
import json

sheetkey = "16N9OApgCo4nrDd4dlpIf0g7962F2mCXImwuhZakSTkg"

gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)

print("Extracting district data from Google Sheets...")

# Load 2022 data
ws_22_26 = sh.worksheet('22->26')
data_22_26 = ws_22_26.get_all_values()

# Load 2024 data
ws_24_26 = sh.worksheet('24->26')
data_24_26 = ws_24_26.get_all_values()

# Load Mi Hazánk multiplier
ws_mi_hazank = sh.worksheet('Mi Hazánk 22')
mi_hazank_data = ws_mi_hazank.get_all_values()

# Parse 2022 data
data_22 = {
    'national': {
        'fidesz': float(data_22_26[1][6].replace(',', '.')),
        'opposition': float(data_22_26[1][7].replace(',', '.')),
        'others': float(data_22_26[1][8].replace(',', '.'))
    },
    'districts': []
}

for i in range(2, len(data_22_26)):
    if data_22_26[i][0] and data_22_26[i][0] != 'Maďarsko':
        try:
            data_22['districts'].append({
                'id': data_22_26[i][0],
                'fidesz': float(data_22_26[i][6].replace(',', '.')),
                'opposition': float(data_22_26[i][7].replace(',', '.')),
                'others': float(data_22_26[i][8].replace(',', '.'))
            })
        except (ValueError, IndexError):
            pass

# Parse 2024 data
data_24 = {
    'national': {
        'fidesz': float(data_24_26[1][1].replace(',', '.')),
        'tisza': float(data_24_26[1][2].replace(',', '.')),
        'others': float(data_24_26[1][3].replace(',', '.'))
    },
    'districts': []
}

for i in range(2, len(data_24_26)):
    if data_24_26[i][0] and data_24_26[i][0] != 'Maďarsko':
        try:
            data_24['districts'].append({
                'id': data_24_26[i][0],
                'fidesz': float(data_24_26[i][1].replace(',', '.')),
                'tisza': float(data_24_26[i][2].replace(',', '.')),
                'others': float(data_24_26[i][3].replace(',', '.'))
            })
        except (ValueError, IndexError):
            pass

# Parse Mi Hazánk multiplier
mi_hazank_multiplier = float(mi_hazank_data[1][3].replace(',', '.'))

# Save to JSON files
with open('hu-2026/district_data_2022.json', 'w', encoding='utf-8') as f:
    json.dump(data_22, f, indent=2, ensure_ascii=False)

with open('hu-2026/district_data_2024.json', 'w', encoding='utf-8') as f:
    json.dump(data_24, f, indent=2, ensure_ascii=False)

with open('hu-2026/mi_hazank_multiplier.json', 'w', encoding='utf-8') as f:
    json.dump({'multiplier': mi_hazank_multiplier}, f, indent=2)

print(f"✓ Saved {len(data_22['districts'])} districts from 2022 to district_data_2022.json")
print(f"✓ Saved {len(data_24['districts'])} districts from 2024 to district_data_2024.json")
print(f"✓ Saved Mi Hazánk multiplier ({mi_hazank_multiplier}) to mi_hazank_multiplier.json")
print("\nData extraction complete!")
