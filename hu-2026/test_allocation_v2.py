"""
Test the Hungarian seat allocation function V2 with values from the Google Sheet.
This version implements the Mi Hazánk <5% rule.
"""

from allocate_seats_hungary_v2 import allocate_seats_hungary_v2, HungarianSeatAllocatorV2

print("=== TESTING HUNGARIAN SEAT ALLOCATION FUNCTION V2 ===\n")

# Test with the values from the sheet
# Input: P2:S2 = Fidesz=35%, Tisza=55%, Others=10%, Mi Hazánk=6.4%
# Expected output: P7:R7 = Fidesz=52, Tisza=140, Mi Hazánk=7

print("Test case from Google Sheet:")
print("Input: Fidesz=35%, Tisza=55%, Others=10%, Mi Hazánk=6.4%")
print("Expected: Fidesz=52, Tisza=140, Mi Hazánk=7")
print()

print("Loading district data from local JSON files...")
allocator = HungarianSeatAllocatorV2()
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
    party_list_votes = allocator.calculate_party_list_votes(35, 55, 6.4)
    
    print("Detailed breakdown:")
    print(f"  District seats: Fidesz={district_seats['fidesz']}, Tisza={district_seats['tisza']}, Others={district_seats['others']}")
    print(f"  Party list votes: Fidesz={party_list_votes['fidesz']}, Tisza={party_list_votes['tisza']}, Mi Hazánk={party_list_votes['mi_hazank']}")

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

print("=== TESTING Mi Hazánk <5% RULE ===\n")

# Test with Mi Hazánk below 5%
print("Test case: Fidesz=40%, Tisza=50%, Others=6%, Mi Hazánk=4% (< 5%)")
f, t, m = allocator.allocate_seats(40, 50, 6, 4)
print(f"Result: Fidesz={f}, Tisza={t}, Mi Hazánk={m}, Total={f+t+m}")
print(f"Mi Hazánk seats: {m} (should be 0 since Mi Hazánk < 5%)")
if m == 0:
    print("✓ Mi Hazánk <5% rule working correctly")
else:
    print("✗ Mi Hazánk <5% rule NOT working - Mi Hazánk should have 0 seats")
print()

# Test with Mi Hazánk at exactly 5%
print("Test case: Fidesz=40%, Tisza=50%, Others=5%, Mi Hazánk=5.0% (= 5%)")
f, t, m = allocator.allocate_seats(40, 50, 5, 5.0)
print(f"Result: Fidesz={f}, Tisza={t}, Mi Hazánk={m}, Total={f+t+m}")
print(f"Mi Hazánk seats: {m} (should be > 0 since Mi Hazánk >= 5%)")
if m > 0:
    print("✓ Mi Hazánk >=5% rule working correctly")
else:
    print("✗ Mi Hazánk >=5% rule NOT working - Mi Hazánk should have seats")
print()

# Test with Mi Hazánk above 5%
print("Test case: Fidesz=35%, Tisza=55%, Others=3.4%, Mi Hazánk=6.6% (> 5%)")
f, t, m = allocator.allocate_seats(35, 55, 3.4, 6.6)
print(f"Result: Fidesz={f}, Tisza={t}, Mi Hazánk={m}, Total={f+t+m}")
print(f"Mi Hazánk seats: {m} (should be > 0 since Mi Hazánk > 5%)")
if m > 0:
    print("✓ Mi Hazánk >5% rule working correctly")
else:
    print("✗ Mi Hazánk >5% rule NOT working - Mi Hazánk should have seats")
print()

print("=== TESTING CONVENIENCE FUNCTION ===\n")

# Test the convenience function (this will load data from local JSON files)
print("Using convenience function allocate_seats_hungary_v2()...")
f, t, m = allocate_seats_hungary_v2(35, 55, 10, 6.4)
print(f"Result: Fidesz={f}, Tisza={t}, Mi Hazánk={m}")
print()

print("=== ALL TESTS COMPLETE ===")
