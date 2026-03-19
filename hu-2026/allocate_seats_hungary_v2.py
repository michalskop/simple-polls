"""
Hungarian seat allocation function for 2026 elections - Version 2.

This function replicates the exact calculation from the Google Sheet:
https://docs.google.com/spreadsheets/d/16N9OApgCo4nrDd4dlpIf0g7962F2mCXImwuhZakSTkg/

Key differences from v1:
- Follows the exact Google Sheet formulas
- Implements the Mi Hazánk <5% rule: if Mi Hazánk is below 5%, it's excluded from 
  list mandate calculation, and only Fidesz and Tisza compete for the 93 list seats
- Uses local JSON data files (no Google Sheets API required)

The Hungarian electoral system has:
- 106 single-member districts (first-past-the-post)
- 93 party list seats (proportional, D'Hondt method)
- Total: 199 seats

Input: National polling percentages for Fidesz, Tisza, Others, Mi Hazánk
Output: Seat counts for Fidesz, Tisza, Mi Hazánk
"""

import json
import os
import numpy as np


class HungarianSeatAllocatorV2:
    """
    Class to handle Hungarian seat allocation calculations using exact Google Sheet formulas.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the allocator.
        
        Args:
            data_dir: Directory containing the district data JSON files.
                     If None, uses the same directory as this script.
        """
        if data_dir is None:
            data_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.data_dir = data_dir
        self._load_district_data()
    
    def _load_district_data(self):
        """Load historical district data from local JSON files."""
        with open(os.path.join(self.data_dir, 'district_data_2022.json'), 'r', encoding='utf-8') as f:
            data_22 = json.load(f)
        
        self.national_22 = data_22['national']
        self.district_data_22 = data_22['districts']
        
        with open(os.path.join(self.data_dir, 'district_data_2024.json'), 'r', encoding='utf-8') as f:
            data_24 = json.load(f)
        
        self.national_24 = data_24['national']
        self.district_data_24 = data_24['districts']
        
        with open(os.path.join(self.data_dir, 'mi_hazank_multiplier.json'), 'r', encoding='utf-8') as f:
            mi_hazank_data = json.load(f)
        
        self.mi_hazank_multiplier = mi_hazank_data['multiplier']
        
        with open(os.path.join(self.data_dir, 'district_valid_votes.json'), 'r', encoding='utf-8') as f:
            valid_votes_data = json.load(f)
        
        self.district_valid_votes = valid_votes_data['districts']
        self.overseas_votes = valid_votes_data['overseas']
        
        with open(os.path.join(self.data_dir, 'mi_hazank_district_votes.json'), 'r', encoding='utf-8') as f:
            mi_hazank_data = json.load(f)
        
        self.mi_hazank_national_2022 = mi_hazank_data['national_pct_2022']
        self.mi_hazank_district_votes = mi_hazank_data['districts']
    
    def project_district_from_22(self, district_22, fidesz_target, tisza_target, others_target):
        """
        Project a district's 2026 results from 2022 data.
        Formula: district_2026 = (district_2022 / national_2022) * national_2026
        Then normalize to 100%.
        """
        scaled_f = (district_22['fidesz'] / self.national_22['fidesz']) * fidesz_target
        scaled_t = (district_22['opposition'] / self.national_22['opposition']) * tisza_target
        scaled_o = (district_22['others'] / self.national_22['others']) * others_target
        
        total = scaled_f + scaled_t + scaled_o
        return {
            'fidesz': (scaled_f / total) * 100,
            'tisza': (scaled_t / total) * 100,
            'others': (scaled_o / total) * 100
        }
    
    def project_district_from_24(self, district_24, fidesz_target, tisza_target, others_target):
        """
        Project a district's 2026 results from 2024 data.
        Formula: district_2026 = (district_2024 / national_2024) * national_2026
        Then normalize to 100%.
        """
        scaled_f = (district_24['fidesz'] / self.national_24['fidesz']) * fidesz_target
        scaled_t = (district_24['tisza'] / self.national_24['tisza']) * tisza_target
        scaled_o = (district_24['others'] / self.national_24['others']) * others_target
        
        total = scaled_f + scaled_t + scaled_o
        return {
            'fidesz': (scaled_f / total) * 100,
            'tisza': (scaled_t / total) * 100,
            'others': (scaled_o / total) * 100
        }
    
    def calculate_district_seats(self, fidesz_pct, tisza_pct, others_pct):
        """
        Calculate district (FPTP) seats based on national polling.
        
        Returns:
            Dict with 'fidesz', 'tisza', 'others' district seat counts
        """
        fidesz_wins = 0
        tisza_wins = 0
        others_wins = 0
        
        for i in range(len(self.district_data_22)):
            proj_22 = self.project_district_from_22(
                self.district_data_22[i], 
                fidesz_pct, tisza_pct, others_pct
            )
            
            proj_24 = self.project_district_from_24(
                self.district_data_24[i],
                fidesz_pct, tisza_pct, others_pct
            )
            
            f = (proj_22['fidesz'] + proj_24['fidesz']) / 2
            t = (proj_22['tisza'] + proj_24['tisza']) / 2
            o = (proj_22['others'] + proj_24['others']) / 2
            
            if f > t and f > o:
                fidesz_wins += 1
            elif t > f and t > o:
                tisza_wins += 1
            elif o > f and o > t:
                others_wins += 1
        
        return {
            'fidesz': fidesz_wins,
            'tisza': tisza_wins,
            'others': others_wins
        }
    
    def calculate_party_list_votes(self, fidesz_pct, tisza_pct, mi_hazank_pct):
        """
        Calculate party list votes following the exact Google Sheet formula.
        
        Party list votes = District votes + Residuals + Overseas votes
        
        Residual formula (from Google Sheet):
        - For Fidesz vs Tisza two-party race in each district:
          - If Tisza > Fidesz: Fidesz residual = all Fidesz votes, Tisza residual = (Tisza - Fidesz)
          - If Fidesz > Tisza: Fidesz residual = (Fidesz - Tisza), Tisza residual = all Tisza votes
        - Mi Hazánk residual = all Mi Hazánk votes (they don't win districts)
        
        Returns:
            Dict with 'fidesz', 'tisza', 'mi_hazank' party list vote counts
        """
        fidesz_district_total = 0
        tisza_district_total = 0
        mi_hazank_district_total = 0
        
        fidesz_residual = 0
        tisza_residual = 0
        mi_hazank_residual = 0
        
        others_pct = 100 - fidesz_pct - tisza_pct - mi_hazank_pct
        
        for i in range(len(self.district_data_22)):
            valid_votes = self.district_valid_votes[i]['valid_votes']
            
            proj_22 = self.project_district_from_22(
                self.district_data_22[i], 
                fidesz_pct, tisza_pct, others_pct
            )
            proj_24 = self.project_district_from_24(
                self.district_data_24[i],
                fidesz_pct, tisza_pct, others_pct
            )
            
            f_pct = (proj_22['fidesz'] + proj_24['fidesz']) / 2 / 100
            t_pct = (proj_22['tisza'] + proj_24['tisza']) / 2 / 100
            
            f_votes = round(valid_votes * f_pct)
            t_votes = round(valid_votes * t_pct)
            
            mi_hazank_multiplier = mi_hazank_pct / self.mi_hazank_national_2022
            mi_votes = round(self.mi_hazank_district_votes[i]['votes_2022'] * mi_hazank_multiplier)
            
            fidesz_district_total += f_votes
            tisza_district_total += t_votes
            mi_hazank_district_total += mi_votes
            
            if t_votes > f_votes:
                fidesz_residual += f_votes
                tisza_residual += (t_votes - f_votes)
            else:
                fidesz_residual += (f_votes - t_votes)
                tisza_residual += t_votes
            
            mi_hazank_residual += mi_votes
        
        overseas_total = 320000
        overseas_fidesz = overseas_total * 0.9
        overseas_tisza = overseas_total * 0.08
        overseas_mi_hazank = overseas_total * 0.02
        
        fidesz_votes = round(fidesz_district_total + fidesz_residual + overseas_fidesz)
        tisza_votes = round(tisza_district_total + tisza_residual + overseas_tisza)
        mi_hazank_votes = round(mi_hazank_district_total + mi_hazank_residual + overseas_mi_hazank)
        
        return {
            'fidesz': fidesz_votes,
            'tisza': tisza_votes,
            'mi_hazank': mi_hazank_votes
        }
    
    def allocate_party_list_seats_dhondt(self, votes_dict, total_seats=93):
        """
        Allocate party list seats using D'Hondt method.
        
        Args:
            votes_dict: Dict with party names as keys and vote counts as values
            total_seats: Total number of seats to allocate (default 93)
        
        Returns:
            Dict with party names as keys and seat counts as values
        """
        seats = {party: 0 for party in votes_dict}
        
        for _ in range(total_seats):
            quotients = {
                party: votes_dict[party] / (seats[party] + 1)
                for party in votes_dict
            }
            
            winner = max(quotients, key=quotients.get)
            seats[winner] += 1
        
        return seats
    
    def allocate_seats(self, fidesz_pct, tisza_pct, others_pct, mi_hazank_pct):
        """
        Main function to allocate all seats.
        
        Implements the special rule: If Mi Hazánk < 5%, exclude it from list seat
        calculation and allocate all 93 list seats between Fidesz and Tisza only.
        
        Args:
            fidesz_pct: Fidesz national polling percentage
            tisza_pct: Tisza national polling percentage
            others_pct: Others national polling percentage
            mi_hazank_pct: Mi Hazánk national polling percentage
        
        Returns:
            Tuple of (fidesz_seats, tisza_seats, mi_hazank_seats)
        """
        district_seats = self.calculate_district_seats(fidesz_pct, tisza_pct, others_pct)
        
        party_list_votes = self.calculate_party_list_votes(fidesz_pct, tisza_pct, mi_hazank_pct)
        
        if mi_hazank_pct < 5.0:
            votes_for_allocation = {
                'fidesz': party_list_votes['fidesz'],
                'tisza': party_list_votes['tisza']
            }
            party_list_seats = self.allocate_party_list_seats_dhondt(votes_for_allocation, total_seats=93)
            party_list_seats['mi_hazank'] = 0
        else:
            party_list_seats = self.allocate_party_list_seats_dhondt(party_list_votes, total_seats=93)
        
        fidesz_total = district_seats['fidesz'] + party_list_seats['fidesz']
        tisza_total = district_seats['tisza'] + party_list_seats['tisza']
        mi_hazank_total = party_list_seats['mi_hazank']
        
        return (fidesz_total, tisza_total, mi_hazank_total)


def allocate_seats_hungary_v2(fidesz_pct, tisza_pct, others_pct, mi_hazank_pct, allocator=None):
    """
    Convenience function to allocate Hungarian parliamentary seats (Version 2).
    
    Args:
        fidesz_pct: Fidesz national polling percentage (0-100)
        tisza_pct: Tisza national polling percentage (0-100)
        others_pct: Others national polling percentage (0-100)
        mi_hazank_pct: Mi Hazánk national polling percentage (0-100)
        allocator: Optional pre-initialized HungarianSeatAllocatorV2 instance
    
    Returns:
        Tuple of (fidesz_seats, tisza_seats, mi_hazank_seats)
    
    Example:
        >>> seats = allocate_seats_hungary_v2(35, 55, 10, 6.4)
        >>> print(f"Fidesz: {seats[0]}, Tisza: {seats[1]}, Mi Hazánk: {seats[2]}")
    """
    if allocator is None:
        allocator = HungarianSeatAllocatorV2()
    
    return allocator.allocate_seats(fidesz_pct, tisza_pct, others_pct, mi_hazank_pct)
