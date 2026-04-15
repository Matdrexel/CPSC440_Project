"""
basic_game.py - Introduction to Infinite Texas Hold'em.

This is a truthful demo of the phase-1 engine:
- cards are dealt with replacement,
- new hand types exist,
- straights beat flushes,
- no blocker effects exist,
- pot settlement is zero-sum.
"""

from __future__ import annotations

import os
import sys
from collections import Counter


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from infinite_holdem import Card, InfiniteDeck, InfiniteHandEvaluator, InfiniteHoldemGame


def _duplicate_strings(cards):
    counts = Counter(str(card) for card in cards)
    return [card for card, count in counts.items() if count > 1]


def demonstrate_replacement_dealing() -> None:
    print("=" * 60)
    print("DEALING WITH REPLACEMENT")
    print("=" * 60)

    deck = InfiniteDeck(seed=42)
    print("\nDealing five separate 7-card samples:")
    for sample_index in range(5):
        cards = deck.deal(7)
        duplicates = _duplicate_strings(cards)
        duplicate_label = f" <- duplicates: {duplicates}" if duplicates else ""
        print(f"  Sample {sample_index + 1}: {[str(card) for card in cards]}{duplicate_label}")

    print("\nEach draw is independent, so duplicates are valid in this variant.")


def demonstrate_new_hand_types() -> None:
    print("\n" + "=" * 60)
    print("NEW HAND TYPES")
    print("=" * 60)

    evaluator = InfiniteHandEvaluator()
    examples = {
        "Flush Five": [Card.from_string("As")] * 5,
        "Five of a Kind": [
            Card.from_string("Kh"),
            Card.from_string("Ks"),
            Card.from_string("Kd"),
            Card.from_string("Kc"),
            Card.from_string("Kh"),
        ],
        "Flush House": [
            Card.from_string("Qh"),
            Card.from_string("Qh"),
            Card.from_string("Qh"),
            Card.from_string("9h"),
            Card.from_string("9h"),
        ],
    }

    for label, cards in examples.items():
        result = evaluator.evaluate(cards)
        print(f"\n{label}:")
        print(f"  Cards: {cards}")
        print(f"  Result: {result.description}")


def demonstrate_ranking_changes() -> None:
    print("\n" + "=" * 60)
    print("RANKING CHANGES")
    print("=" * 60)

    evaluator = InfiniteHandEvaluator()
    straight = evaluator.evaluate(
        [
            Card.from_string("Jc"),
            Card.from_string("Td"),
            Card.from_string("9h"),
            Card.from_string("8s"),
            Card.from_string("7c"),
        ]
    )
    flush = evaluator.evaluate(
        [
            Card.from_string("Ah"),
            Card.from_string("Kh"),
            Card.from_string("9h"),
            Card.from_string("5h"),
            Card.from_string("2h"),
        ]
    )

    print(f"Straight: {straight.description}")
    print(f"Flush: {flush.description}")
    if straight > flush:
        print("\nStraight beats flush in Infinite Hold'em because straights are rarer.")


def demonstrate_no_blockers() -> None:
    print("\n" + "=" * 60)
    print("NO BLOCKER EFFECTS")
    print("=" * 60)
    print(
        """
In standard poker, your cards remove combinations from the opponent's range.
In infinite hold'em, draws are with replacement:
  - if you hold As, the opponent can still hold As,
  - your cards do not change the probability of their private cards,
  - opponent modeling becomes purely behavioral.
""".strip()
    )


def play_sample_hand() -> None:
    print("\n" + "=" * 60)
    print("PLAYING A SAMPLE HAND")
    print("=" * 60)

    game = InfiniteHoldemGame(
        num_players=2,
        starting_stack=10000,
        small_blind=50,
        big_blind=100,
        seed=12345,
    )
    game.deal_hole_cards()

    print(f"\nBlinds posted. Pot: {game.pot}")
    print(f"Stacks: {game.stacks}")
    print("\n--- PREFLOP ---")
    print(f"Player 0: {game.hole_cards[0]}")
    print(f"Player 1: {game.hole_cards[1]}")

    hole_duplicates = _duplicate_strings(game.hole_cards[0] + game.hole_cards[1])
    if hole_duplicates:
        print(f"  Duplicate hole cards across players: {hole_duplicates}")

    game.apply_action(0, "raise", 300)
    print("Player 0 raises to 300")
    game.apply_action(1, "call")
    print("Player 1 calls")

    if game.is_betting_round_complete():
        game.advance_street()

    print("\n--- FLOP ---")
    print(f"Board: {game.board}")
    game.apply_action(1, "check")
    print("Player 1 checks")
    game.apply_action(0, "raise", 400)
    print("Player 0 bets 400")
    game.apply_action(1, "call")
    print("Player 1 calls")

    if game.is_betting_round_complete():
        game.advance_street()

    print("\n--- TURN ---")
    print(f"Board: {game.board}")
    game.apply_action(1, "check")
    print("Player 1 checks")
    game.apply_action(0, "check")
    print("Player 0 checks")

    if game.is_betting_round_complete():
        game.advance_street()

    print("\n--- RIVER ---")
    print(f"Board: {game.board}")
    game.apply_action(1, "check")
    print("Player 1 checks")
    game.apply_action(0, "check")
    print("Player 0 checks")

    print("\n--- SHOWDOWN ---")
    winners, results = game.determine_winner()
    for player_idx, result in enumerate(results):
        if result is not None:
            print(f"Player {player_idx}: {result.description}")

    board_duplicates = _duplicate_strings(game.hole_cards[0] + game.hole_cards[1] + game.board)
    if board_duplicates:
        print(f"Duplicate cards across the full hand: {board_duplicates}")

    winners, payoffs = game.settle_pot()
    print(f"\nWinners: {winners}")
    print(f"Final stacks: {game.stacks}")
    print(f"Net payoffs: {payoffs}")
    print(f"Payoff sum: {sum(payoffs)}")


def demonstrate_fold_settlement() -> None:
    print("\n" + "=" * 60)
    print("FOLD SETTLEMENT")
    print("=" * 60)

    game = InfiniteHoldemGame(num_players=2, starting_stack=1000, seed=999)
    game.deal_hole_cards()

    print(f"\nAfter blinds - Pot: {game.pot}, Stacks: {game.stacks}")
    print(f"Player 0: {game.hole_cards[0]}")
    print(f"Player 1: {game.hole_cards[1]}")

    game.apply_action(0, "raise", 200)
    print("Player 0 raises to 200")
    game.apply_action(1, "fold")
    print("Player 1 folds")

    winners, payoffs = game.settle_pot()
    print(f"\nWinner: Player {winners[0]}")
    print(f"Final stacks: {game.stacks}")
    print(f"Net payoffs: {payoffs}")
    print(f"Payoff sum: {sum(payoffs)}")


if __name__ == "__main__":
    demonstrate_replacement_dealing()
    demonstrate_new_hand_types()
    demonstrate_ranking_changes()
    demonstrate_no_blockers()
    play_sample_hand()
    demonstrate_fold_settlement()
