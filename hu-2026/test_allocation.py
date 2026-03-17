"""
Test the Hungarian seat allocation function with values from the Google Sheet.
"""

from allocate_seats_hungary import allocate_seats_hungary, HungarianSeatAllocator

print("=== TESTING HUNGARIAN SEAT ALLOCATION FUNCTION ===\n")

# Test with the values from the sheet
# Input: P2:S2 = Fidesz=35%, Tisza=55%, Others=10%, Mi Hazánk=6.4%
# Expected output: P7:R7 = Fidesz=52, Tisza=140, Mi Hazánk=7

print("Test case from Google Sheet:")
print("Input: Fidesz=35%, Tisza=55%, Others=10%, Mi Hazánk=6.4%")
print("Expected: Fidesz=52, Tisza=140, Mi Hazánk=7")
print()

print("Loading district data from local JSON files...")
allocator = HungarianSeatAllocator()
print(f"Loaded {len(allocator.district_data_22)} districts from 2022 data")
print(f"Loaded {len(allocator.district_data_24)} districts from 2024 data")
print()

print("Calculating seats...")
fidesz_seats, tisza_seats, mi_hazank_seats = allocator.allocate_seats(35, 55, 10, 6.4)

print(f"\nResult: Fidesz={fidesz_seats}, Tisza={tisza_seats}, Mi Hazánk={mi_hazank_seats}")
print()

# Check if results match
if fidesz_seats == 52 and tisza_seats == 140 and mi_hazank_seats == 7:
    print("✓ TEST PASSED! Results match the Google Sheet.")
else:
    print("✗ TEST FAILED! Results do not match.")
    print(f"  Expected: Fidesz=52, Tisza=140, Mi Hazánk=7")
    print(f"  Got:      Fidesz={fidesz_seats}, Tisza={tisza_seats}, Mi Hazánk={mi_hazank_seats}")
    print()
    
    # Show detailed breakdown
    district_seats = allocator.calculate_district_seats(35, 55, 10)
    party_list_seats = allocator.calculate_party_list_seats(35, 55, 6.4, district_seats)
    
    print("Detailed breakdown:")
    print(f"  District seats: Fidesz={district_seats['fidesz']}, Tisza={district_seats['tisza']}, Others={district_seats['others']}")
    print(f"  Party list seats: Fidesz={party_list_seats['fidesz']}, Tisza={party_list_seats['tisza']}, Mi Hazánk={party_list_seats['mi_hazank']}")
    print(f"  Total: Fidesz={district_seats['fidesz'] + party_list_seats['fidesz']}, Tisza={district_seats['tisza'] + party_list_seats['tisza']}, Mi Hazánk={party_list_seats['mi_hazank']}")

print("\n=== TESTING WITH DIFFERENT VALUES ===\n")

# Test with some different polling numbers
test_cases = [
    (35, 55, 10, 6.4, 52, 140, 7),
    (40, 50, 10, 5, 78, 115, 6),
    (30, 60, 10, 7, 38, 154, 7),
    (45, 45, 10, 8, 101, 89, 9),
]

for fidesz, tisza, others, mi_hazank, exp_f, exp_t, exp_m in test_cases:
    print(f"Input: Fidesz={fidesz}%, Tisza={tisza}%, Others={others}%, Mi Hazánk={mi_hazank}%")
    f, t, m = allocator.allocate_seats(fidesz, tisza, others, mi_hazank)
    print(f"Result:   Fidesz={f}, Tisza={t}, Mi Hazánk={m}, Total={f+t+m}")
    print(f"Expected: Fidesz={exp_f}, Tisza={exp_t}, Mi Hazánk={exp_m}, Total={exp_f+exp_t+exp_m}")
    
    if f == exp_f and t == exp_t and m == exp_m:
        print("✓ Match")
    else:
        diff_f = f - exp_f
        diff_t = t - exp_t
        diff_m = m - exp_m
        print(f"Difference: F{diff_f:+d}, T{diff_t:+d}, MH{diff_m:+d}")
    print()

print("=== TESTING CONVENIENCE FUNCTION ===\n")

# Test the convenience function (this will load data from local JSON files)
print("Using convenience function allocate_seats_hungary()...")
f, t, m = allocate_seats_hungary(35, 55, 10, 6.4)
print(f"Result: Fidesz={f}, Tisza={t}, Mi Hazánk={m}")
