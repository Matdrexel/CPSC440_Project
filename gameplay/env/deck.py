import random
from env.card import rank, suit
 
# All 52 possible card values (0–51)
ALL_CARDS = list(range(52))
 
RANK_NAMES = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
              9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
SUIT_NAMES = {0: '♣', 1: '♦', 2: '♥', 3: '♠'}
 
 
def deal_card() -> int:
    """Draw a single card at random (with replacement)."""
    return random.choice(ALL_CARDS)
 
 
def deal_cards(n: int) -> list[int]:
    """Draw n cards with replacement."""
    return [deal_card() for _ in range(n)]
 
 
def card_name(c: int) -> str:
    """Human-readable card string, e.g. 'A♠', 'T♥'."""
    return f"{RANK_NAMES[rank(c)]}{SUIT_NAMES[suit(c)]}"
 
 
def hand_name(cards: list[int]) -> str:
    """Human-readable representation of a list of cards."""
    return ' '.join(card_name(c) for c in cards)