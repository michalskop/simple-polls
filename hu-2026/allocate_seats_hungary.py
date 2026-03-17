"""
Hungarian seat allocation function for 2026 elections.

This function replicates the calculation from the Google Sheet:
https://docs.google.com/spreadsheets/d/16N9OApgCo4nrDd4dlpIf0g7962F2mCXImwuhZakSTkg/

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
import pandas as pd


class HungarianSeatAllocator:
    """
    Class to handle Hungarian seat allocation calculations.
    Loads historical district data once and reuses it for multiple calculations.
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
        self.district_data_22 = None
        self.district_data_24 = None
        self.mi_hazank_multiplier = None
        
        self._load_district_data()
    
    def _load_district_data(self):
        """Load historical district data from local JSON files."""
        # Load 2022 data
        with open(os.path.join(self.data_dir, 'district_data_2022.json'), 'r', encoding='utf-8') as f:
            data_22 = json.load(f)
        
        self.national_22 = data_22['national']
        self.district_data_22 = data_22['districts']
        
        # Load 2024 data
        with open(os.path.join(self.data_dir, 'district_data_2024.json'), 'r', encoding='utf-8') as f:
            data_24 = json.load(f)
        
        self.national_24 = data_24['national']
        self.district_data_24 = data_24['districts']
        
        # Load Mi Hazánk multiplier
        with open(os.path.join(self.data_dir, 'mi_hazank_multiplier.json'), 'r', encoding='utf-8') as f:
            mi_hazank_data = json.load(f)
        
        self.mi_hazank_multiplier = mi_hazank_data['multiplier']
        
        # Load district valid votes
        with open(os.path.join(self.data_dir, 'district_valid_votes.json'), 'r', encoding='utf-8') as f:
            valid_votes_data = json.load(f)
        
        self.district_valid_votes = valid_votes_data['districts']
        self.overseas_votes = valid_votes_data['overseas']
        
        # Load Mi Hazánk historical vote data
        with open(os.path.join(self.data_dir, 'mi_hazank_district_votes.json'), 'r', encoding='utf-8') as f:
            mi_hazank_data = json.load(f)
        
        self.mi_hazank_national_2022 = mi_hazank_data['national_pct_2022']
        self.mi_hazank_district_votes = mi_hazank_data['districts']
    
    def project_district_from_22(self, district_22, fidesz_target, tisza_target, others_target):
        """
        Project a district's 2026 results from 2022 data.
        
        Args:
            district_22: Dict with 'fidesz', 'opposition', 'others' percentages
            fidesz_target, tisza_target, others_target: National 2026 polling percentages
        
        Returns:
            Dict with projected 'fidesz', 'tisza', 'others' percentages
        """
        # Scale each party proportionally
        scaled_f = (district_22['fidesz'] / self.national_22['fidesz']) * fidesz_target
        scaled_t = (district_22['opposition'] / self.national_22['opposition']) * tisza_target
        scaled_o = (district_22['others'] / self.national_22['others']) * others_target
        
        # Normalize to 100%
        total = scaled_f + scaled_t + scaled_o
        return {
            'fidesz': (scaled_f / total) * 100,
            'tisza': (scaled_t / total) * 100,
            'others': (scaled_o / total) * 100
        }
    
    def project_district_from_24(self, district_24, fidesz_target, tisza_target, others_target):
        """
        Project a district's 2026 results from 2024 data.
        
        Args:
            district_24: Dict with 'fidesz', 'tisza', 'others' percentages
            fidesz_target, tisza_target, others_target: National 2026 polling percentages
        
        Returns:
            Dict with projected 'fidesz', 'tisza', 'others' percentages
        """
        # Scale each party proportionally
        scaled_f = (district_24['fidesz'] / self.national_24['fidesz']) * fidesz_target
        scaled_t = (district_24['tisza'] / self.national_24['tisza']) * tisza_target
        scaled_o = (district_24['others'] / self.national_24['others']) * others_target
        
        # Normalize to 100%
        total = scaled_f + scaled_t + scaled_o
        return {
            'fidesz': (scaled_f / total) * 100,
            'tisza': (scaled_t / total) * 100,
            'others': (scaled_o / total) * 100
        }
    
    def calculate_district_seats(self, fidesz_pct, tisza_pct, others_pct):
        """
        Calculate district (FPTP) seats based on national polling.
        
        Args:
            fidesz_pct, tisza_pct, others_pct: National polling percentages
        
        Returns:
            Dict with 'fidesz', 'tisza', 'others' district seat counts
        """
        fidesz_wins = 0
        tisza_wins = 0
        others_wins = 0
        
        # Project each district and determine winner
        for i in range(len(self.district_data_22)):
            # Project from 2022
            proj_22 = self.project_district_from_22(
                self.district_data_22[i], 
                fidesz_pct, tisza_pct, others_pct
            )
            
            # Project from 2024
            proj_24 = self.project_district_from_24(
                self.district_data_24[i],
                fidesz_pct, tisza_pct, others_pct
            )
            
            # Average the two projections
            f = (proj_22['fidesz'] + proj_24['fidesz']) / 2
            t = (proj_22['tisza'] + proj_24['tisza']) / 2
            o = (proj_22['others'] + proj_24['others']) / 2
            
            # Determine winner (FPTP)
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
    
    def calculate_party_list_seats(self, fidesz_pct, tisza_pct, mi_hazank_pct, 
                                   district_winners, total_party_list_seats=93):
        """
        Calculate party list seats using D'Hondt method.
        
        The party list votes are calculated from:
        1. District-level vote counts (valid votes × vote share)
        2. Residual votes (votes for parties that didn't win the district)
        3. Overseas votes (fixed amounts)
        
        Args:
            fidesz_pct, tisza_pct, mi_hazank_pct: National polling percentages
            district_winners: List of district winners ('fidesz', 'tisza', or 'others')
            total_party_list_seats: Total party list seats to allocate (default 93)
        
        Returns:
            Dict with 'fidesz', 'tisza', 'mi_hazank' party list seat counts
        """
        # Calculate party list votes from district-level data
        # Formula from Google Sheet:
        # - District votes = vote_share * valid_votes
        # - Residual: Winner gets (their_votes - opponent_votes), Loser gets all their votes
        # - Party list = District votes + Residuals + Overseas
        
        fidesz_district_total = 0
        tisza_district_total = 0
        mi_hazank_district_total = 0
        
        fidesz_residual = 0
        tisza_residual = 0
        mi_hazank_residual = 0
        
        for i in range(len(self.district_data_22)):
            # Get valid votes for this district
            valid_votes = self.district_valid_votes[i]['valid_votes']
            
            # Project district vote shares
            proj_22 = self.project_district_from_22(
                self.district_data_22[i], 
                fidesz_pct, tisza_pct, 100 - fidesz_pct - tisza_pct - mi_hazank_pct
            )
            proj_24 = self.project_district_from_24(
                self.district_data_24[i],
                fidesz_pct, tisza_pct, 100 - fidesz_pct - tisza_pct - mi_hazank_pct
            )
            
            # Average the projections to get vote shares (as percentages)
            f_pct = (proj_22['fidesz'] + proj_24['fidesz']) / 2 / 100
            t_pct = (proj_22['tisza'] + proj_24['tisza']) / 2 / 100
            
            # Calculate district vote counts
            # Round to integers to match Google Sheet's integer arithmetic
            f_votes = round(valid_votes * f_pct)
            t_votes = round(valid_votes * t_pct)
            # Mi Hazánk uses historical 2022 votes scaled by the ratio of current/historical percentages
            mi_hazank_multiplier = mi_hazank_pct / self.mi_hazank_national_2022
            mi_votes = round(self.mi_hazank_district_votes[i]['votes_2022'] * mi_hazank_multiplier)
            
            # Accumulate district totals
            fidesz_district_total += f_votes
            tisza_district_total += t_votes
            mi_hazank_district_total += mi_votes
            
            # Calculate residuals based on Google Sheet formula:
            # Fidesz residual: IF(Tisza > Fidesz, Fidesz, Fidesz - Tisza)
            # Tisza residual: IF(Fidesz > Tisza, Tisza, Tisza - Fidesz)
            # Mi Hazánk residual: always all their votes
            
            if t_votes > f_votes:
                # Tisza won
                fidesz_residual += f_votes  # Loser gets all votes
                tisza_residual += (t_votes - f_votes)  # Winner gets margin
            else:
                # Fidesz won
                fidesz_residual += (f_votes - t_votes)  # Winner gets margin
                tisza_residual += t_votes  # Loser gets all votes
            
            mi_hazank_residual += mi_votes  # Always all votes
        
        # Overseas votes: 320,000 total split as 90% Fidesz, 8% Tisza, 2% Mi Hazánk
        overseas_total = 320000
        overseas_fidesz = overseas_total * 0.9
        overseas_tisza = overseas_total * 0.08
        overseas_mi_hazank = overseas_total * 0.02
        
        # Party list total = District votes + Residuals + Overseas
        # Round to integers to match Google Sheet precision
        fidesz_votes = round(fidesz_district_total + fidesz_residual + overseas_fidesz)
        tisza_votes = round(tisza_district_total + tisza_residual + overseas_tisza)
        mi_hazank_votes = round(mi_hazank_district_total + mi_hazank_residual + overseas_mi_hazank)
        
        # D'Hondt method
        seats = {'fidesz': 0, 'tisza': 0, 'mi_hazank': 0}
        votes = {'fidesz': fidesz_votes, 'tisza': tisza_votes, 'mi_hazank': mi_hazank_votes}
        
        for _ in range(total_party_list_seats):
            # Calculate quotients for each party
            quotients = {
                party: votes[party] / (seats[party] + 1)
                for party in votes
            }
            
            # Award seat to party with highest quotient
            winner = max(quotients, key=quotients.get)
            seats[winner] += 1
        
        return seats
    
    def allocate_seats(self, fidesz_pct, tisza_pct, others_pct, mi_hazank_pct):
        """
        Main function to allocate all seats.
        
        Args:
            fidesz_pct: Fidesz national polling percentage
            tisza_pct: Tisza national polling percentage
            others_pct: Others national polling percentage
            mi_hazank_pct: Mi Hazánk national polling percentage
        
        Returns:
            Tuple of (fidesz_seats, tisza_seats, mi_hazank_seats)
        """
        # Step 1: Calculate district seats
        district_seats = self.calculate_district_seats(fidesz_pct, tisza_pct, others_pct)
        
        # Step 2: Calculate party list seats (calculates district winners internally)
        party_list_seats = self.calculate_party_list_seats(
            fidesz_pct, tisza_pct, mi_hazank_pct, None
        )
        
        # Step 3: Sum up total seats
        fidesz_total = district_seats['fidesz'] + party_list_seats['fidesz']
        tisza_total = district_seats['tisza'] + party_list_seats['tisza']
        mi_hazank_total = party_list_seats['mi_hazank']  # Mi Hazánk only gets party list seats
        
        return (fidesz_total, tisza_total, mi_hazank_total)


def allocate_seats_hungary(fidesz_pct, tisza_pct, others_pct, mi_hazank_pct, allocator=None):
    """
    Convenience function to allocate Hungarian parliamentary seats.
    
    Args:
        fidesz_pct: Fidesz national polling percentage (0-100)
        tisza_pct: Tisza national polling percentage (0-100)
        others_pct: Others national polling percentage (0-100)
        mi_hazank_pct: Mi Hazánk national polling percentage (0-100)
        allocator: Optional pre-initialized HungarianSeatAllocator instance
                  (for better performance when calling multiple times)
    
    Returns:
        Tuple of (fidesz_seats, tisza_seats, mi_hazank_seats)
    
    Example:
        >>> seats = allocate_seats_hungary(35, 55, 10, 6.4)
        >>> print(f"Fidesz: {seats[0]}, Tisza: {seats[1]}, Mi Hazánk: {seats[2]}")
        Fidesz: 52, Tisza: 140, Mi Hazánk: 7
    """
    if allocator is None:
        allocator = HungarianSeatAllocator()
    
    return allocator.allocate_seats(fidesz_pct, tisza_pct, others_pct, mi_hazank_pct)
