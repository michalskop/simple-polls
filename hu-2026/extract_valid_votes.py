"""
Extract valid vote counts for each district to enable proper party list calculation.
"""

import gspread
import json

sheetkey = "16N9OApgCo4nrDd4dlpIf0g7962F2mCXImwuhZakSTkg"

gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)

# Read the 'party list' tab which has valid votes per district
ws_party_list = sh.worksheet('party list')
party_list_data = ws_party_list.get_all_values()

print("Extracting valid vote counts for each district...")

# Parse district valid votes
district_valid_votes = []

for i in range(2, len(party_list_data)):
    if party_list_data[i][0] and party_list_data[i][0] != 'Maďarsko':
        try:
            district_id = party_list_data[i][0]
            # Column G (index 6) has "Počet platných" (valid votes)
            valid_votes_str = party_list_data[i][6].replace('\xa0', '').replace(' ', '')
            if valid_votes_str:
                valid_votes = int(valid_votes_str)
                district_valid_votes.append({
                    'id': district_id,
                    'valid_votes': valid_votes
                })
        except (ValueError, IndexError):
            pass

print(f"Extracted valid votes for {len(district_valid_votes)} districts")

# Also get overseas votes
# Row 3 of 'party list' tab, columns T:V (indices 19:22)
overseas_row_idx = 2  # Row 3 (0-indexed)
if len(party_list_data) > overseas_row_idx:
    # Check if this is the overseas row
    if 'Zahraničí' in party_list_data[overseas_row_idx][18]:
        overseas_fidesz = int(party_list_data[overseas_row_idx][19].replace('\xa0', '').replace(' ', ''))
        overseas_tisza = int(party_list_data[overseas_row_idx][20].replace('\xa0', '').replace(' ', ''))
        overseas_mi_hazank = int(party_list_data[overseas_row_idx][21].replace('\xa0', '').replace(' ', ''))
        
        overseas_votes = {
            'fidesz': overseas_fidesz,
            'tisza': overseas_tisza,
            'mi_hazank': overseas_mi_hazank
        }
        
        print(f"Overseas votes: Fidesz={overseas_fidesz}, Tisza={overseas_tisza}, Mi Hazánk={overseas_mi_hazank}")
    else:
        print("Warning: Could not find overseas votes row")
        overseas_votes = {'fidesz': 288000, 'tisza': 25600, 'mi_hazank': 6400}
else:
    print("Warning: Could not find overseas votes")
    overseas_votes = {'fidesz': 288000, 'tisza': 25600, 'mi_hazank': 6400}

# Save to JSON
with open('hu-2026/district_valid_votes.json', 'w', encoding='utf-8') as f:
    json.dump({
        'districts': district_valid_votes,
        'overseas': overseas_votes
    }, f, indent=2, ensure_ascii=False)

print(f"✓ Saved valid votes to district_valid_votes.json")
print("\nSample districts:")
for d in district_valid_votes[:5]:
    print(f"  {d['id']}: {d['valid_votes']:,} valid votes")
