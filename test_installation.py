#!/usr/bin/env python3
"""
test_installation.py - Verify runtime and dev dependencies for phase 1.
"""

from __future__ import annotations

import importlib.util
import sys


def test_python_version() -> bool:
    print("Checking Python version...")
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 11):
        print(f"  ✓ Python {major}.{minor} (3.11+ required)")
        return True
    print(f"  ✗ Python {major}.{minor} (need 3.11+)")
    return False


def test_numpy() -> bool:
    print("\nChecking NumPy...")
    try:
        import numpy as np

        print(f"  ✓ numpy {np.__version__}")
        return True
    except Exception as exc:
        print(f"  ✗ {exc}")
        return False


def test_pytest() -> bool:
    print("\nChecking pytest...")
    if importlib.util.find_spec("pytest") is None:
        print("  ✗ pytest not installed")
        return False
    print("  ✓ pytest available")
    return True


def test_pokerkit() -> bool:
    print("\nChecking PokerKit (reference only)...")
    try:
        from pokerkit import Automation, NoLimitTexasHoldem

        state = NoLimitTexasHoldem.create_state(
            (Automation.BLIND_OR_STRADDLE_POSTING,),
            True,
            0,
            (50, 100),
            100,
            (1000, 1000),
            2,
        )
        print("  ✓ PokerKit imported")
        print(f"  ✓ Standard NLHE state created (pot after blinds: {state.total_pot_amount})")
        return True
    except Exception as exc:
        print(f"  ✗ {exc}")
        return False


def test_openspiel() -> bool:
    print("\nChecking OpenSpiel...")
    try:
        import pyspiel
        from open_spiel.python.algorithms import cfr

        game = pyspiel.load_game("kuhn_poker")
        _ = cfr.CFRSolver(game)
        print("  ✓ pyspiel imported")
        print("  ✓ CFR solver available")
        return True
    except Exception as exc:
        print(f"  ✗ {exc}")
        return False


def test_infinite_holdem() -> bool:
    print("\nChecking Infinite Hold'em engine...")
    try:
        from infinite_holdem import Card, HandRank, InfiniteDeck, InfiniteHandEvaluator, InfiniteHoldemGame

        deck = InfiniteDeck(seed=42)
        dealt_cards = deck.deal(200)
        duplicate_deal = len({str(card) for card in dealt_cards}) < len(dealt_cards)
        if not duplicate_deal:
            print("  ✗ duplicate dealing check failed")
            return False
        print("  ✓ replacement dealing can produce duplicate cards")

        evaluator = InfiniteHandEvaluator()
        five_of_a_kind = evaluator.evaluate(
            [
                Card.from_string("As"),
                Card.from_string("Ah"),
                Card.from_string("Ad"),
                Card.from_string("Ac"),
                Card.from_string("As"),
            ]
        )
        if five_of_a_kind.rank != HandRank.FIVE_OF_A_KIND:
            print(f"  ✗ expected Five of a Kind, got {five_of_a_kind.description}")
            return False
        print("  ✓ Five of a Kind recognized")

        straight = evaluator.evaluate([Card.from_string(card) for card in ("9h", "8d", "7c", "6s", "5h")])
        flush = evaluator.evaluate([Card.from_string(card) for card in ("Kh", "Jh", "8h", "4h", "2h")])
        if not straight > flush:
            print("  ✗ straight should beat flush in infinite rules")
            return False
        print("  ✓ straight beats flush")

        fold_game = InfiniteHoldemGame(seed=7)
        fold_game.deal_hole_cards()
        fold_game.bet_or_raise(0, 200)
        fold_game.fold(1)
        fold_winners, fold_payoffs = fold_game.settle_pot()
        if fold_winners != [0] or sum(fold_payoffs) != 0:
            print(f"  ✗ fold settlement incorrect: winners={fold_winners}, payoffs={fold_payoffs}")
            return False
        print("  ✓ fold settlement is zero-sum")

        showdown_game = InfiniteHoldemGame(seed=8)
        showdown_game.deal_hole_cards()
        showdown_game.call(0)
        showdown_game.check(1)
        showdown_game.advance_street()
        showdown_game.check(1)
        showdown_game.check(0)
        showdown_game.advance_street()
        showdown_game.check(1)
        showdown_game.check(0)
        showdown_game.advance_street()
        showdown_game.check(1)
        showdown_game.check(0)
        showdown_winners, showdown_payoffs = showdown_game.settle_pot()
        if sum(showdown_payoffs) != 0:
            print(
                f"  ✗ showdown settlement incorrect: winners={showdown_winners}, payoffs={showdown_payoffs}"
            )
            return False
        print("  ✓ showdown settlement is zero-sum")

        return True
    except Exception as exc:
        print(f"  ✗ {exc}")
        return False


def run_quick_demo() -> bool:
    print("\n" + "=" * 50)
    print("INFINITE HOLD'EM DEMO")
    print("=" * 50)

    try:
        from infinite_holdem import InfiniteHoldemGame

        game = InfiniteHoldemGame(seed=999)
        game.deal_hole_cards()
        print(f"\nPlayer 0: {game.hole_cards[0]}")
        print(f"Player 1: {game.hole_cards[1]}")

        game.call(0)
        game.check(1)
        game.advance_street()
        print(f"Flop: {game.board}")
        game.check(1)
        game.check(0)
        game.advance_street()
        print(f"Turn: {game.board}")
        game.check(1)
        game.check(0)
        game.advance_street()
        print(f"River: {game.board}")
        game.check(1)
        game.check(0)

        winners, payoffs = game.settle_pot()
        print(f"Winner(s): {winners}")
        print(f"Payoffs: {payoffs}")
        return True
    except Exception as exc:
        print(f"Demo failed: {exc}")
        return False


def main() -> bool:
    print("=" * 50)
    print("INFINITE TEXAS HOLD'EM - INSTALLATION TEST")
    print("=" * 50)
    print("(Cards dealt with replacement variant)")

    results = [
        ("Python", test_python_version()),
        ("NumPy", test_numpy()),
        ("pytest", test_pytest()),
        ("PokerKit", test_pokerkit()),
        ("OpenSpiel", test_openspiel()),
        ("Infinite Hold'em", test_infinite_holdem()),
    ]

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    all_passed = all(passed for _, passed in results)
    for name, passed in results:
        print(f"  {'✓' if passed else '✗'} {name}")

    if all_passed:
        print("\nAll required runtime and dev components are available.")
        run_quick_demo()
    else:
        print("\nSome required components are still missing.")
        print("The most common phase-1 blocker is missing pytest in the active venv.")

    return all_passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
