"""
simple_agent.py - AI agents for Infinite Texas Hold'em.

The agents in this module rely on the canonical runtime engine in
`infinite_holdem.py` for action legality, street advancement, and pot
settlement.
"""

from __future__ import annotations

import os
import random
import sys
from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Optional, Tuple


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from infinite_holdem import InfiniteDeck, InfiniteHandEvaluator, InfiniteHoldemGame, Card, HandRank


class BaseAgent(ABC):
    """Abstract base class for Infinite Hold'em agents."""

    def __init__(self, name: str = "Agent"):
        self.name = name
        self.evaluator = InfiniteHandEvaluator()

    @abstractmethod
    def get_action(self, game: InfiniteHoldemGame, player_idx: int) -> Tuple[str, Optional[int]]:
        """Return `(action, total_to_amount_or_none)`."""


class RandomAgent(BaseAgent):
    """Completely random baseline agent."""

    def __init__(self):
        super().__init__("RandomAgent")

    def get_action(self, game: InfiniteHoldemGame, player_idx: int) -> Tuple[str, Optional[int]]:
        legal_actions = game.get_legal_actions(player_idx)
        if not legal_actions:
            return "check", None

        chosen_action = random.choice(legal_actions)
        if chosen_action != "raise":
            return chosen_action, None

        min_raise_to = game.get_min_raise_to(player_idx)
        max_raise_to = game.get_max_raise_to(player_idx)
        if min_raise_to is None or max_raise_to is None:
            return "check", None
        return "raise", random.randint(min_raise_to, max_raise_to)


class CallStationAgent(BaseAgent):
    """Never folds and never raises."""

    def __init__(self):
        super().__init__("CallStation")

    def get_action(self, game: InfiniteHoldemGame, player_idx: int) -> Tuple[str, Optional[int]]:
        if "check" in game.get_legal_actions(player_idx):
            return "check", None
        return "call", None


class TightAggressiveAgent(BaseAgent):
    """Simple TAG-style agent adapted to infinite hold'em."""

    def __init__(self, aggression: float = 0.6):
        super().__init__("TAG")
        self.aggression = aggression

    def estimate_hand_strength(self, game: InfiniteHoldemGame, player_idx: int) -> float:
        hole = game.hole_cards[player_idx]
        board = game.board
        if not board:
            return self._preflop_strength(hole)

        hand_result = self.evaluator.evaluate(hole + board)
        rank_strengths = {
            HandRank.HIGH_CARD: 0.10,
            HandRank.PAIR: 0.25,
            HandRank.TWO_PAIR: 0.45,
            HandRank.THREE_OF_A_KIND: 0.55,
            HandRank.FULL_HOUSE: 0.70,
            HandRank.FLUSH: 0.75,
            HandRank.STRAIGHT: 0.80,
            HandRank.FOUR_OF_A_KIND: 0.90,
            HandRank.FIVE_OF_A_KIND: 0.95,
            HandRank.FLUSH_HOUSE: 0.97,
            HandRank.STRAIGHT_FLUSH: 0.98,
            HandRank.FLUSH_FIVE: 0.99,
        }
        return rank_strengths.get(hand_result.rank, 0.10)

    def _preflop_strength(self, hole: List[Card]) -> float:
        first_rank = hole[0].rank_value
        second_rank = hole[1].rank_value
        suited = hole[0].suit == hole[1].suit
        paired = hole[0].rank == hole[1].rank

        if hole[0] == hole[1]:
            return 0.40 + (first_rank / 26)
        if paired:
            return 0.50 + (first_rank / 24)

        high = max(first_rank, second_rank)
        low = min(first_rank, second_rank)
        strength = (high + low) / 24
        if suited:
            strength += 0.08
        if 0 < high - low <= 4:
            strength += 0.02
        return min(0.80, strength)

    def get_action(self, game: InfiniteHoldemGame, player_idx: int) -> Tuple[str, Optional[int]]:
        legal_actions = game.get_legal_actions(player_idx)
        strength = self.estimate_hand_strength(game, player_idx)
        to_call = game.get_to_call(player_idx)
        pot = game.pot

        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
        if "raise" in legal_actions and strength > 0.65:
            min_raise_to = game.get_min_raise_to(player_idx)
            max_raise_to = game.get_max_raise_to(player_idx)
            if min_raise_to is not None and max_raise_to is not None:
                target = int(pot * (0.5 + 0.5 * self.aggression))
                return "raise", max(min_raise_to, min(target, max_raise_to))

        if strength > 0.35 or strength > pot_odds:
            if "check" in legal_actions:
                return "check", None
            if "call" in legal_actions:
                return "call", None

        if "raise" in legal_actions and random.random() < 0.1 * self.aggression:
            min_raise_to = game.get_min_raise_to(player_idx)
            max_raise_to = game.get_max_raise_to(player_idx)
            if min_raise_to is not None and max_raise_to is not None:
                target = int(max(game.big_blind, pot * 0.6))
                return "raise", max(min_raise_to, min(target, max_raise_to))

        if "check" in legal_actions:
            return "check", None
        if "fold" in legal_actions:
            return "fold", None
        return "call", None


class EquityMonteCarloAgent(BaseAgent):
    """Agent that estimates equity with replacement-dealing Monte Carlo."""

    def __init__(self, simulations: int = 100, aggression: float = 0.5):
        super().__init__("EquityMC")
        self.simulations = simulations
        self.aggression = aggression

    def estimate_equity(self, game: InfiniteHoldemGame, player_idx: int) -> float:
        deck = InfiniteDeck()
        my_hole = game.hole_cards[player_idx]
        board = game.board
        wins = 0.0

        for _ in range(self.simulations):
            opponent_hole = deck.deal(2)
            future_board = deck.deal(5 - len(board)) if len(board) < 5 else []
            full_board = board + future_board
            my_hand = self.evaluator.evaluate(my_hole + full_board)
            opponent_hand = self.evaluator.evaluate(opponent_hole + full_board)

            if my_hand > opponent_hand:
                wins += 1.0
            elif my_hand == opponent_hand:
                wins += 0.5

        return wins / self.simulations

    def get_action(self, game: InfiniteHoldemGame, player_idx: int) -> Tuple[str, Optional[int]]:
        legal_actions = game.get_legal_actions(player_idx)
        equity = self.estimate_equity(game, player_idx)
        to_call = game.get_to_call(player_idx)
        pot = game.pot
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0

        if "raise" in legal_actions and equity > 0.70:
            min_raise_to = game.get_min_raise_to(player_idx)
            max_raise_to = game.get_max_raise_to(player_idx)
            if min_raise_to is not None and max_raise_to is not None:
                target = int(pot * (0.6 + 0.4 * equity))
                return "raise", max(min_raise_to, min(target, max_raise_to))

        if equity > pot_odds:
            if "check" in legal_actions:
                if "raise" in legal_actions and equity > 0.55 and random.random() < self.aggression:
                    min_raise_to = game.get_min_raise_to(player_idx)
                    max_raise_to = game.get_max_raise_to(player_idx)
                    if min_raise_to is not None and max_raise_to is not None:
                        target = int(max(game.big_blind, pot * 0.5))
                        return "raise", max(min_raise_to, min(target, max_raise_to))
                return "check", None
            if "call" in legal_actions:
                return "call", None

        if "raise" in legal_actions and random.random() < 0.15 * self.aggression:
            min_raise_to = game.get_min_raise_to(player_idx)
            max_raise_to = game.get_max_raise_to(player_idx)
            if min_raise_to is not None and max_raise_to is not None:
                target = int(max(game.big_blind, pot * 0.5))
                return "raise", max(min_raise_to, min(target, max_raise_to))

        if "check" in legal_actions:
            return "check", None
        if "fold" in legal_actions:
            return "fold", None
        return "call", None


def _normalize_action(
    game: InfiniteHoldemGame,
    player_idx: int,
    action: str,
    amount: Optional[int],
) -> Tuple[str, Optional[int]]:
    legal_actions = game.get_legal_actions(player_idx)
    if not legal_actions:
        raise ValueError(f"No legal actions available for player {player_idx}")

    if action == "raise" and "raise" in legal_actions:
        min_raise_to = game.get_min_raise_to(player_idx)
        max_raise_to = game.get_max_raise_to(player_idx)
        if min_raise_to is None or max_raise_to is None:
            action = "check" if "check" in legal_actions else "call"
            return action, None
        if amount is None:
            amount = min_raise_to
        return "raise", max(min_raise_to, min(amount, max_raise_to))

    if action in legal_actions:
        return action, None

    for fallback in ("check", "call", "fold"):
        if fallback in legal_actions:
            return fallback, None

    return legal_actions[0], None


def _print_board(game: InfiniteHoldemGame, previous_street: str) -> str:
    if game.street != previous_street and game.street in {"flop", "turn", "river"}:
        print(f"\n--- {game.street.upper()} ---")
        print(f"Board: {game.board}")
    return game.street


def play_hand(
    agent0: BaseAgent,
    agent1: BaseAgent,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> List[float]:
    """Play one heads-up hand using the canonical engine."""

    game = InfiniteHoldemGame(
        num_players=2,
        starting_stack=10000,
        small_blind=50,
        big_blind=100,
        seed=seed,
    )
    game.deal_hole_cards()
    agents = [agent0, agent1]

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"{agent0.name} vs {agent1.name}")
        print(f"{'=' * 50}")
        print(f"Player 0: {game.hole_cards[0]}")
        print(f"Player 1: {game.hole_cards[1]}")
        all_hole_cards = game.hole_cards[0] + game.hole_cards[1]
        card_counts = Counter(str(card) for card in all_hole_cards)
        duplicates = [card for card, count in card_counts.items() if count > 1]
        if duplicates:
            print(f"  -> Duplicate cards: {duplicates} (valid in infinite hold'em)")
        print("\n--- PREFLOP ---")

    logged_street = "preflop"

    while not game.is_hand_over():
        if game.actor_index is None:
            if game.is_betting_round_complete():
                game.advance_street()
                if verbose:
                    logged_street = _print_board(game, logged_street)
                continue
            raise RuntimeError("Engine reached a non-terminal state with no current actor")

        actor = game.actor_index
        action, amount = agents[actor].get_action(game, actor)
        action, amount = _normalize_action(game, actor, action, amount)

        if verbose:
            action_label = action if amount is None else f"{action} to {amount}"
            print(f"{agents[actor].name}: {action_label}")

        game.apply_action(actor, action, amount)

    winners, hand_results = game.determine_winner()
    winners, payoffs = game.settle_pot()

    if verbose:
        if game.street == "showdown":
            print("\n--- SHOWDOWN ---")
            for player_idx, result in enumerate(hand_results):
                if result is not None:
                    print(f"Player {player_idx}: {result.description}")

        if len(winners) == 1:
            print(f"\nWinner: {agents[winners[0]].name}")
        else:
            split_names = ", ".join(agents[player].name for player in winners)
            print(f"\nSplit pot: {split_names}")
        print(f"Payoffs: {payoffs}")
        print(f"Final stacks: {game.stacks}")
        print(f"Payoff sum: {sum(payoffs)}")

    return payoffs


def run_tournament(agents: List[BaseAgent], num_hands: int = 500) -> None:
    """Run a simple round-robin tournament."""

    print(f"\n{'=' * 60}")
    print("INFINITE HOLD'EM TOURNAMENT")
    print(f"{'=' * 60}")
    print(f"Agents: {[agent.name for agent in agents]}")
    print(f"Hands per matchup: {num_hands}")

    results = {agent.name: 0.0 for agent in agents}

    for left_index, agent0 in enumerate(agents):
        for right_index, agent1 in enumerate(agents):
            if left_index >= right_index:
                continue

            print(f"\n{agent0.name} vs {agent1.name}...", end=" ")
            total0 = 0.0
            total1 = 0.0

            for hand_index in range(num_hands):
                seed = (left_index + 1) * 100000 + (right_index + 1) * 1000 + hand_index
                payoff0, payoff1 = play_hand(agent0, agent1, seed=seed)
                total0 += payoff0
                total1 += payoff1

            average0 = total0 / num_hands
            average1 = total1 / num_hands
            print(f"Avg: {average0:+.1f} vs {average1:+.1f}")

            results[agent0.name] += average0
            results[agent1.name] += average1

    print(f"\n{'=' * 60}")
    print("FINAL STANDINGS")
    print(f"{'=' * 60}")
    for name, score in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name}: {score:+.1f}")


if __name__ == "__main__":
    print("INFINITE TEXAS HOLD'EM - AGENT TESTING")
    print("(Cards dealt WITH REPLACEMENT)\n")

    random_agent = RandomAgent()
    call_station = CallStationAgent()
    tag_agent = TightAggressiveAgent(aggression=0.6)
    equity_agent = EquityMonteCarloAgent(simulations=50, aggression=0.5)

    print("=== SAMPLE HANDS ===")
    for offset in range(3):
        play_hand(tag_agent, equity_agent, verbose=True, seed=offset * 100)

    print("\n=== VERIFYING POT SETTLEMENT ===")
    verification_payoffs = play_hand(tag_agent, random_agent, verbose=True, seed=42)
    print(f"Zero-sum check: {sum(verification_payoffs)}")

    run_tournament([random_agent, call_station, tag_agent, equity_agent], num_hands=200)
