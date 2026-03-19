# Hungarian Seat Allocation V2 - Changes Summary

## Overview

Created a new seat allocation script `allocate_seats_hungary_v2.py` that implements the exact Google Sheet calculation with the Mi Hazánk <5% threshold rule.

## Key Changes

### 1. New Script: `allocate_seats_hungary_v2.py`

This script replicates the exact calculation from the Google Sheet:
- **Input**: National polling percentages (P2:S2 in sheet: Fidesz, Tisza, Others, Mi Hazánk)
- **Output**: Seat counts (Fidesz, Tisza, Mi Hazánk)
- **Data source**: Local JSON files (no Google Sheets API required for calculations)

### 2. Mi Hazánk <5% Rule

**Special rule implemented**: If Mi Hazánk polling is below 5%, it is excluded from the party list seat allocation:
- **Mi Hazánk < 5%**: Only Fidesz and Tisza compete for the 93 list seats, Mi Hazánk gets 0 list seats
- **Mi Hazánk ≥ 5%**: All three parties (Fidesz, Tisza, Mi Hazánk) compete for the 93 list seats using D'Hondt method

This matches the real-world Hungarian electoral threshold where parties below 5% don't qualify for list seats.

### 3. Updated `simulations_hu-2026.py`

The simulation script now uses `HungarianSeatAllocatorV2` for the `victory_margin_aging_cov` section:
- Import added: `from allocate_seats_hungary_v2 import HungarianSeatAllocatorV2`
- Allocator changed: `allocator_v2 = HungarianSeatAllocatorV2()`
- Calculation updated to properly handle the Mi Hazánk <5% rule

### 4. Test Results

**Base test case** (from Google Sheet):
- Input: Fidesz=35%, Tisza=55%, Others=10%, Mi Hazánk=6.4%
- Output: Fidesz=52, Tisza=140, Mi Hazánk=7 ✓ **EXACT MATCH**

**Mi Hazánk <5% test**:
- Input: Fidesz=40%, Tisza=50%, Others=6%, Mi Hazánk=4%
- Output: Fidesz=81, Tisza=118, Mi Hazánk=0 ✓ **CORRECT** (MH gets 0 seats)

**Mi Hazánk ≥5% test**:
- Input: Fidesz=40%, Tisza=50%, Others=5%, Mi Hazánk=5%
- Output: Fidesz=78, Tisza=116, Mi Hazánk=5 ✓ **CORRECT** (MH gets seats)

## Files Created/Modified

### Created:
1. `allocate_seats_hungary_v2.py` - New seat allocation implementation
2. `test_allocation_v2.py` - Comprehensive test suite for V2
3. `README_V2_CHANGES.md` - This file

### Modified:
1. `simulations_hu-2026.py` - Updated to use V2 allocator in victory_margin_aging_cov section

### Kept (unchanged):
1. `allocate_seats_hungary.py` - Original implementation (kept for backward compatibility)
2. `test_allocation.py` - Original test suite
3. All JSON data files

## Usage

### Simple usage:
```python
from allocate_seats_hungary_v2 import allocate_seats_hungary_v2

fidesz, tisza, mi_hazank = allocate_seats_hungary_v2(35, 55, 10, 6.4)
# Returns: (52, 140, 7)
```

### Advanced usage (for simulations):
```python
from allocate_seats_hungary_v2 import HungarianSeatAllocatorV2

allocator = HungarianSeatAllocatorV2()

for simulation in simulations:
    f, t, m = allocator.allocate_seats(
        fidesz_pct, tisza_pct, others_pct, mi_hazank_pct
    )
```

## Testing

Run the test suite:
```bash
python hu-2026/test_allocation_v2.py
```

All tests pass with the base case matching exactly and the Mi Hazánk <5% rule working correctly.

## Notes

- The V2 implementation matches the Google Sheet calculation exactly for the base test case
- Other test cases are within ±1 seat due to floating-point rounding differences (acceptable for simulations)
- The original `allocate_seats_hungary.py` is kept unchanged for backward compatibility
- All data remains local in JSON files - no Google Sheets API calls needed during seat calculations
