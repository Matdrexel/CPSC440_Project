"""
hand_evaluator.py — Custom Texas Hold'em hand evaluator.

Hand rankings (high to low):
  11 — Flush Five    : 5 identical cards (same rank AND same suit)
  10 — Straight Flush: 5 sequential cards of the same suit
   9 — Flush House   : Full house where all 5 cards share the same suit
   8 — Five of a Kind: 5 cards of the same rank (any suits)
   7 — Four of a Kind
   6 — Straight      (promoted above flush in this ruleset)
   5 — Flush
   4 — Full House    (demoted below flush in this ruleset)
   3 — Three of a Kind
   2 — Two Pair
   1 — One Pair
   0 — High Card

Tuple comparison: higher first element wins; ties broken by subsequent elements.
All tiebreaker masks use bit-positions equal to card rank (Ace = bit 14).
"""

from env.card import rank, suit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Masks for straights A-high down to 6-high (bits set at each rank position)
STRAIGHT_MASKS = [(0b11111 << i) for i in range(10, 1, -1)]
WHEEL_MASK = (1 << 14) | 0b111100

def find_straight(mask: int) -> int:
    """
    Given a bitmask of ranks present, return the straight mask if one exists,
    else 0.  Returns the highest possible straight.
    """
    for straight in STRAIGHT_MASKS:
        if (mask & straight) == straight:
            return straight
    if (mask & WHEEL_MASK) == WHEEL_MASK:
        return 1  # wheel — lowest straight
    return 0

def top_n_from_mask(mask: int, n: int) -> int:
    """Return a bitmask containing only the n highest set bits of mask."""
    res = 0
    for r in range(14, 1, -1):
        if mask & (1 << r):
            res |= 1 << r
            n -= 1
            if n == 0:
                break
    return res

# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def check_hand(cards: list[int]) -> tuple:
    """
    Evaluate the best 5-card hand from `cards` (typically 7 cards).
 
    Returns a comparable tuple — higher tuple = stronger hand.
    First element is the hand rank (0–11), subsequent elements are
    bitmasks used for tiebreaking.
    """
    # --- Tally ranks and suits ------------------------------------------
    ranks = [0] * 15           # ranks[r] = count of rank r across all cards
    suits = [0] * 4            # suits[s] = count of suit s across all cards
    suit_rank_counts = {}      # [suit][rank] = count
    suit_masks = [0] * 4       # suit_masks[s] = bitmask of ranks in suit s
    rank_mask = 0              # bitmask of all ranks present (any count)
    suit_pairs_mask = [0] * 4  # suit_pairs_mask[s] = bitmask of all suited pairs present (any count)
    suit_trips_mask = [0] * 4  # suit_pairs_mask[s] = bitmask of all suited trips present (any count)
    flush5_mask = 0            # bitmask of all flush 5 ranks present (any count)

    for c in cards:
        r = rank(c)
        s = suit(c)
        ranks[r] += 1
        suits[s] += 1
        if (c in suit_rank_counts):
            suit_rank_counts[c] += 1
            suit_pairs_mask[s] |= 1 << r
            if suit_rank_counts[c] >= 5:
                flush5_mask |= 1 << r
            elif suit_rank_counts[c] >= 3:
                suit_trips_mask[s] |= 1 << r
        else:
            suit_rank_counts[c] = 1
        suit_masks[s] |= 1 << r
        rank_mask |= 1 << r

    # ====================================================================
    # 11 — Flush Five: 5 identical cards (same rank AND suit)
    # ====================================================================
    if flush5_mask:
        best = top_n_from_mask(flush5_mask, 1)
        return (11, best)

    # --- Global suit count tallies -------------------------------------------
    straight_flush = None
    flush_house = None
    flush = None

    for s in range(4):
        if suits[s] >= 5:
            sf = find_straight(suit_masks[s])
            if sf:
                if straight_flush is None or sf > straight_flush:
                    straight_flush = sf
                continue

            trip_mask = suit_trips_mask[s]
            pair_mask = suit_pairs_mask[s]
            if trip_mask.bit_count() >= 1 and pair_mask.bit_count() >= 2:
                trip = top_n_from_mask(trip_mask, 1)
                pair = top_n_from_mask(pair_mask - trip, 1)
                if flush_house is None or (trip, pair) > flush_house:
                    flush_house = (trip, pair)
                continue
            
            new_flush = []
            suit_mask = suit_masks[s]
            while len(new_flush) < 5 and suit_mask != 0:
                top = top_n_from_mask(suit_mask, 1)
                if top & trip_mask:
                    new_flush += [top] * 3
                elif top & pair_mask:
                    new_flush += [top] * 2
                else:
                    new_flush.append(top)
                
                suit_mask -= top
            
            new_flush = tuple(new_flush[:5])
            if flush is None or new_flush > flush:
                flush = new_flush
            

    # ====================================================================
    # 10 — Straight Flush
    # ====================================================================
    if straight_flush is not None:
        return (10, straight_flush)

    # ====================================================================
    # 9 — Flush House: full house where all 5 cards share a suit
    # ====================================================================
    if flush_house is not None:
        return (9, *flush_house)
 
    # --- Global rank count tallies -------------------------------------------
    quints_mask = quads_mask = trips_mask = pairs_mask = singles_mask = 0

    for r in range(14, 1, -1):
        c = ranks[r]
        if c >= 5:
            quints_mask |= 1 << r
        elif c == 4:
            quads_mask |= 1 << r
        elif c == 3:
            trips_mask |= 1 << r
        elif c == 2:
            pairs_mask |= 1 << r
        elif c == 1:
            singles_mask |= 1 << r

    # ====================================================================
    # 8 — Five of a Kind
    # ====================================================================
    if quints_mask:
        return (8, top_n_from_mask(quints_mask, 1))

    # ====================================================================
    # 7 — Four of a Kind
    # ====================================================================
    if quads_mask:
        best = top_n_from_mask(quads_mask, 1)
        kicker_pool = rank_mask - best
        return (7, best, top_n_from_mask(kicker_pool, 1))

    # ====================================================================
    # 6 — Straight
    # ====================================================================
    straight = find_straight(rank_mask)
    if straight:
        return (6, straight)

    # ====================================================================
    # 5 — Flush
    # ====================================================================
    if flush is not None:
        return (5, *flush)

    # ====================================================================
    # 4 — Full House  (demoted below flush in this ruleset)
    # ====================================================================
    if trips_mask and (pairs_mask or trips_mask.bit_count() > 1):
        best_trips = top_n_from_mask(trips_mask, 1)
        remainder = trips_mask - best_trips
        best_pair = top_n_from_mask(remainder | pairs_mask, 1)
        return (4, best_trips, best_pair)

    # ====================================================================
    # 3 — Three of a Kind
    # ====================================================================
    if trips_mask:
        return (3, trips_mask, top_n_from_mask(singles_mask, 2))

    # ====================================================================
    # 2 — Two Pair
    # ====================================================================
    if pairs_mask.bit_count() >= 2:
        top_2 = top_n_from_mask(pairs_mask, 2)
        return (2, top_2, top_n_from_mask(rank_mask - top_2, 1))

    # ====================================================================
    # 1 — One Pair
    # ====================================================================
    if pairs_mask:
        return (1, pairs_mask, top_n_from_mask(singles_mask, 3))

    # ====================================================================
    # 0 — High Card
    # ====================================================================
    return (0, top_n_from_mask(singles_mask, 5))


# ---------------------------------------------------------------------------
# Hand name for display
# ---------------------------------------------------------------------------

HAND_NAMES = {
    11: "Flush Five",
    10: "Straight Flush",
     9: "Flush House",
     8: "Five of a Kind",
     7: "Four of a Kind",
     6: "Straight",
     5: "Flush",
     4: "Full House",
     3: "Three of a Kind",
     2: "Two Pair",
     1: "One Pair",
     0: "High Card",
}

def hand_name(hand_tuple: tuple) -> str:
    return HAND_NAMES[hand_tuple[0]]