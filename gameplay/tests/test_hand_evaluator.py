"""
test_hand_evaluator.py — Unit tests for the custom hand evaluator.

Folder structure:
    project/
    ├── env/
    │   ├── card.py
    │   ├── deck.py
    │   └── hand_evaluator.py
    └── tests/
        └── test_hand_evaluator.py

Run from the project root with:  python -m pytest tests/test_hand_evaluator.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env'))

import pytest
from card import card as make_card
from hand_evaluator import check_hand, hand_name

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cards(*args):
    """Build a list of cards from (rank, suit) tuples."""
    return [make_card(r, s) for r, s in args]


def hand_rank(hand_tuple):
    return hand_tuple[0]


# ---------------------------------------------------------------------------
# Flush Five (11) — 5 identical cards
# ---------------------------------------------------------------------------

class TestFlushFive:
    def test_five_aces_of_spades(self):
        # 5x A♠ + 2 other cards
        hand = cards((14,3),(14,3),(14,3),(14,3),(14,3),(2,0),(3,1))
        assert hand_rank(check_hand(hand)) == 11

    def test_five_kings_of_hearts(self):
        hand = cards((13,2),(13,2),(13,2),(13,2),(13,2),(7,0),(9,1))
        assert hand_rank(check_hand(hand)) == 11

    def test_higher_flush_five_beats_lower(self):
        aces = cards((14,3),(14,3),(14,3),(14,3),(14,3),(2,0),(3,1))
        kings = cards((13,2),(13,2),(13,2),(13,2),(13,2),(2,0),(3,1))
        assert check_hand(aces) > check_hand(kings)

    def test_flush_five_beats_straight_flush(self):
        flush5 = cards((14,3),(14,3),(14,3),(14,3),(14,3),(2,0),(3,1))
        sf = cards((10,0),(11,0),(12,0),(13,0),(14,0),(2,1),(3,2))
        assert check_hand(flush5) > check_hand(sf)

    def test_four_identical_not_flush_five(self):
        # Only 4 of same card — should NOT be flush five
        hand = cards((14,3),(14,3),(14,3),(14,3),(2,0),(3,1),(5,2))
        assert hand_rank(check_hand(hand)) != 11


# ---------------------------------------------------------------------------
# Straight Flush (10)
# ---------------------------------------------------------------------------

class TestStraightFlush:
    def test_royal_flush(self):
        hand = cards((10,0),(11,0),(12,0),(13,0),(14,0),(2,1),(3,2))
        assert hand_rank(check_hand(hand)) == 10

    def test_low_straight_flush(self):
        hand = cards((2,1),(3,1),(4,1),(5,1),(6,1),(9,0),(14,2))
        assert hand_rank(check_hand(hand)) == 10

    def test_wheel_straight_flush(self):
        # A-2-3-4-5 same suit
        hand = cards((14,2),(2,2),(3,2),(4,2),(5,2),(9,0),(10,1))
        assert hand_rank(check_hand(hand)) == 10

    def test_straight_flush_beats_flush_house(self):
        sf = cards((10,0),(11,0),(12,0),(13,0),(14,0),(2,1),(3,2))
        # Flush house: trips + pair all same suit
        fh = cards((14,0),(14,0),(14,0),(13,0),(13,0),(2,1),(3,2))
        assert check_hand(sf) > check_hand(fh)


# ---------------------------------------------------------------------------
# Flush House (9)
# ---------------------------------------------------------------------------

class TestFlushHouse:
    def test_basic_flush_house(self):
        # Three Aces of clubs + two Kings of clubs
        hand = cards((14,0),(14,0),(14,0),(13,0),(13,0),(2,1),(3,2))
        assert hand_rank(check_hand(hand)) == 9

    def test_flush_house_beats_five_of_a_kind(self):
        fh = cards((14,0),(14,0),(14,0),(13,0),(13,0),(2,1),(3,2))
        foak5 = cards((7,0),(7,1),(7,2),(7,3),(7,0),(2,0),(3,1))
        assert check_hand(fh) > check_hand(foak5)

    def test_regular_full_house_not_flush_house(self):
        # Trips in clubs, pair in diamonds — NOT flush house
        hand = cards((14,0),(14,0),(14,0),(13,1),(13,1),(2,2),(3,3))
        assert hand_rank(check_hand(hand)) != 9

    def test_higher_trips_wins_flush_house(self):
        fh_aces = cards((14,0),(14,0),(14,0),(13,0),(13,0),(2,1),(3,2))
        fh_kings = cards((13,0),(13,0),(13,0),(12,0),(12,0),(2,1),(3,2))
        assert check_hand(fh_aces) > check_hand(fh_kings)


# ---------------------------------------------------------------------------
# Five of a Kind (8)
# ---------------------------------------------------------------------------

class TestFiveOfAKind:
    def test_five_aces_different_suits(self):
        # 5 aces across different suits (not all same suit)
        hand = cards((14,0),(14,1),(14,2),(14,3),(14,0),(2,0),(3,1))
        assert hand_rank(check_hand(hand)) == 8

    def test_five_of_a_kind_beats_four_of_a_kind(self):
        fives = cards((7,0),(7,1),(7,2),(7,3),(7,0),(2,0),(3,1))
        fours = cards((14,0),(14,1),(14,2),(14,3),(2,0),(3,1),(5,2))
        assert check_hand(fives) > check_hand(fours)

    def test_higher_five_beats_lower(self):
        aces = cards((14,0),(14,1),(14,2),(14,3),(14,0),(2,0),(3,1))
        kings = cards((13,0),(13,1),(13,2),(13,3),(13,0),(2,0),(3,1))
        assert check_hand(aces) > check_hand(kings)


# ---------------------------------------------------------------------------
# Four of a Kind (7)
# ---------------------------------------------------------------------------

class TestFourOfAKind:
    def test_basic_four_of_a_kind(self):
        hand = cards((14,0),(14,1),(14,2),(14,3),(2,0),(3,1),(5,2))
        assert hand_rank(check_hand(hand)) == 7

    def test_higher_quads_wins(self):
        aces = cards((14,0),(14,1),(14,2),(14,3),(2,0),(3,1),(5,2))
        kings = cards((13,0),(13,1),(13,2),(13,3),(2,0),(3,1),(5,2))
        assert check_hand(aces) > check_hand(kings)

    def test_same_quads_kicker_decides(self):
        ace_kicker = cards((7,0),(7,1),(7,2),(7,3),(14,0),(2,1),(3,2))
        two_kicker = cards((7,0),(7,1),(7,2),(7,3),(2,0),(3,1),(4,2))
        assert check_hand(ace_kicker) > check_hand(two_kicker)


# ---------------------------------------------------------------------------
# Straight (6) — promoted above flush
# ---------------------------------------------------------------------------

class TestStraight:
    def test_basic_straight(self):
        hand = cards((9,0),(10,1),(11,2),(12,3),(13,0),(2,1),(3,2))
        assert hand_rank(check_hand(hand)) == 6

    def test_wheel_straight(self):
        hand = cards((14,0),(2,1),(3,2),(4,3),(5,0),(9,1),(10,2))
        assert hand_rank(check_hand(hand)) == 6

    def test_straight_beats_flush(self):
        straight = cards((9,0),(10,1),(11,2),(12,3),(13,0),(2,1),(3,2))
        flush = cards((2,0),(5,0),(7,0),(9,0),(14,0),(3,1),(4,2))
        assert check_hand(straight) > check_hand(flush)

    def test_higher_straight_wins(self):
        high = cards((10,0),(11,1),(12,2),(13,3),(14,0),(2,1),(3,2))
        low  = cards((2,0),(3,1),(4,2),(5,3),(6,0),(9,1),(10,2))
        assert check_hand(high) > check_hand(low)


# ---------------------------------------------------------------------------
# Flush (5) — demoted below straight
# ---------------------------------------------------------------------------

class TestFlush:
    def test_basic_flush(self):
        # 2,5,9,J,A of clubs — no 5-card straight possible
        hand = cards((2,0),(5,0),(11,0),(9,0),(14,0),(6,1),(8,2))
        assert hand_rank(check_hand(hand)) == 5

    def test_flush_beats_full_house(self):
        flush = cards((2,0),(5,0),(7,0),(9,0),(14,0),(3,1),(4,2))
        full_house = cards((14,0),(14,1),(14,2),(13,0),(13,1),(2,2),(3,3))
        assert check_hand(flush) > check_hand(full_house)

    def test_higher_flush_wins(self):
        ace_high = cards((14,0),(10,0),(8,0),(6,0),(2,0),(3,1),(4,2))
        king_high = cards((13,0),(10,0),(8,0),(6,0),(2,0),(3,1),(4,2))
        assert check_hand(ace_high) > check_hand(king_high)


# ---------------------------------------------------------------------------
# Full House (4) — demoted below flush
# ---------------------------------------------------------------------------

class TestFullHouse:
    def test_basic_full_house(self):
        hand = cards((14,0),(14,1),(14,2),(13,0),(13,1),(2,2),(3,3))
        assert hand_rank(check_hand(hand)) == 4

    def test_full_house_beats_three_of_a_kind(self):
        fh = cards((14,0),(14,1),(14,2),(13,0),(13,1),(2,2),(3,3))
        toak = cards((14,0),(14,1),(14,2),(2,0),(3,1),(5,2),(7,3))
        assert check_hand(fh) > check_hand(toak)

    def test_higher_trips_wins_full_house(self):
        aces_full = cards((14,0),(14,1),(14,2),(2,0),(2,1),(3,2),(4,3))
        kings_full = cards((13,0),(13,1),(13,2),(2,0),(2,1),(3,2),(4,3))
        assert check_hand(aces_full) > check_hand(kings_full)


# ---------------------------------------------------------------------------
# Three of a Kind (3)
# ---------------------------------------------------------------------------

class TestThreeOfAKind:
    def test_basic_trips(self):
        hand = cards((7,0),(7,1),(7,2),(2,0),(3,1),(5,2),(9,3))
        assert hand_rank(check_hand(hand)) == 3

    def test_higher_trips_wins(self):
        aces = cards((14,0),(14,1),(14,2),(2,0),(3,1),(5,2),(9,3))
        kings = cards((13,0),(13,1),(13,2),(2,0),(3,1),(5,2),(9,3))
        assert check_hand(aces) > check_hand(kings)


# ---------------------------------------------------------------------------
# Two Pair (2)
# ---------------------------------------------------------------------------

class TestTwoPair:
    def test_basic_two_pair(self):
        hand = cards((14,0),(14,1),(13,0),(13,1),(2,0),(3,1),(5,2))
        assert hand_rank(check_hand(hand)) == 2

    def test_higher_top_pair_wins(self):
        aces_kings = cards((14,0),(14,1),(13,0),(13,1),(2,0),(3,1),(5,2))
        kings_queens = cards((13,0),(13,1),(12,0),(12,1),(2,0),(3,1),(5,2))
        assert check_hand(aces_kings) > check_hand(kings_queens)

    def test_kicker_decides_two_pair(self):
        ace_kicker = cards((13,0),(13,1),(12,0),(12,1),(14,0),(2,1),(3,2))
        two_kicker  = cards((13,0),(13,1),(12,0),(12,1),(2,0),(3,1),(4,2))
        assert check_hand(ace_kicker) > check_hand(two_kicker)


# ---------------------------------------------------------------------------
# One Pair (1)
# ---------------------------------------------------------------------------

class TestOnePair:
    def test_basic_pair(self):
        hand = cards((14,0),(14,1),(2,0),(3,1),(5,2),(7,3),(9,0))
        assert hand_rank(check_hand(hand)) == 1

    def test_higher_pair_wins(self):
        aces = cards((14,0),(14,1),(2,0),(3,1),(5,2),(7,3),(9,0))
        kings = cards((13,0),(13,1),(2,0),(3,1),(5,2),(7,3),(9,0))
        assert check_hand(aces) > check_hand(kings)


# ---------------------------------------------------------------------------
# High Card (0)
# ---------------------------------------------------------------------------

class TestHighCard:
    def test_basic_high_card(self):
        hand = cards((14,0),(2,1),(4,2),(7,3),(9,0),(11,1),(13,2))
        assert hand_rank(check_hand(hand)) == 0

    def test_higher_top_card_wins(self):
        ace_high = cards((14,0),(2,1),(4,2),(7,3),(9,0),(11,1),(12,2))
        king_high = cards((13,0),(2,1),(4,2),(7,3),(9,0),(11,1),(12,2))
        assert check_hand(ace_high) > check_hand(king_high)


# ---------------------------------------------------------------------------
# Hand ordering sanity checks — one representative of each tier
# ---------------------------------------------------------------------------

class TestHandOrdering:
    def make_hands(self):
        return {
            11: cards((14,3),(14,3),(14,3),(14,3),(14,3),(2,0),(3,1)),  # flush five
            10: cards((10,0),(11,0),(12,0),(13,0),(14,0),(2,1),(3,2)),  # straight flush
             9: cards((14,0),(14,0),(14,0),(13,0),(13,0),(2,1),(3,2)),  # flush house
             8: cards((7,0),(7,1),(7,2),(7,3),(7,0),(2,0),(3,1)),       # five oak
             7: cards((14,0),(14,1),(14,2),(14,3),(2,0),(3,1),(5,2)),   # four oak
             6: cards((9,0),(10,1),(11,2),(12,3),(13,0),(2,1),(3,2)),   # straight
             5: cards((2,0),(5,0),(11,0),(9,0),(14,0),(6,1),(8,2)),     # flush
             4: cards((14,0),(14,1),(14,2),(13,0),(13,1),(2,2),(3,3)),  # full house
             3: cards((7,0),(7,1),(7,2),(2,0),(3,1),(5,2),(9,3)),       # trips
             2: cards((14,0),(14,1),(13,0),(13,1),(2,0),(3,1),(5,2)),   # two pair
             1: cards((14,0),(14,1),(2,0),(3,1),(5,2),(7,3),(9,0)),     # pair
             0: cards((14,0),(2,1),(4,2),(7,3),(9,0),(11,1),(13,2)),    # high card
        }

    def test_all_tiers_correctly_ranked(self):
        hands = self.make_hands()
        evaluated = {tier: check_hand(h) for tier, h in hands.items()}
        tiers = sorted(evaluated.keys(), reverse=True)
        for i in range(len(tiers) - 1):
            higher = tiers[i]
            lower = tiers[i + 1]
            assert evaluated[higher] > evaluated[lower], (
                f"Expected {hand_name(evaluated[higher])} > {hand_name(evaluated[lower])}"
            )

    def test_hand_name_returns_string(self):
        hands = self.make_hands()
        for tier, h in hands.items():
            name = hand_name(check_hand(h))
            assert isinstance(name, str) and len(name) > 0


# ---------------------------------------------------------------------------
# Infinite deck edge cases — duplicate cards
# ---------------------------------------------------------------------------

class TestInfiniteDeckEdgeCases:
    def test_two_identical_cards_in_hand(self):
        # Two A♠ among 7 cards — should work without error
        hand = cards((14,3),(14,3),(2,0),(3,1),(5,2),(7,0),(9,1))
        result = check_hand(hand)
        assert hand_rank(result) == 1  # one pair of aces

    def test_six_of_a_kind_best_five(self):
        # 6 aces — should detect flush five or five oak correctly
        hand = cards((14,0),(14,1),(14,2),(14,3),(14,0),(14,1),(2,0))
        result = check_hand(hand)
        assert hand_rank(result) in (8, 11)  # five oak or flush five

    def test_all_seven_same_card(self):
        hand = cards((14,3),(14,3),(14,3),(14,3),(14,3),(14,3),(14,3))
        result = check_hand(hand)
        assert hand_rank(result) == 11  # flush five

    def test_duplicate_cards_straight_flush(self):
        # Repeated cards that still form a straight flush
        hand = cards((10,0),(11,0),(12,0),(13,0),(14,0),(14,0),(10,0))
        result = check_hand(hand)
        assert hand_rank(result) == 10


# ---------------------------------------------------------------------------
# Duplicate-rank tiebreaker tests
# ---------------------------------------------------------------------------

class TestDuplicateRankTiebreakers:

    # --- Flush (rank tuple needed — duplicates possible within a suit) ----

    def test_flush_aa_beats_flush_ak(self):
        flush_aa = cards((14,0),(14,0),(12,0),(11,0),(10,0),(2,1),(3,2))
        flush_ak = cards((14,0),(13,0),(4,0),(3,0),(2,0),(7,1),(8,2))
        assert check_hand(flush_aa) > check_hand(flush_ak)

    def test_flush_aak_beats_flush_aaq(self):
        flush_aak = cards((14,0),(14,0),(13,0),(2,0),(3,0),(7,1),(8,2))
        flush_aaq = cards((14,0),(14,0),(12,0),(2,0),(3,0),(7,1),(8,2))
        assert check_hand(flush_aak) > check_hand(flush_aaq)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])