"""
simulate_games.py - Generate training data for Infinite Texas Hold'em.

This script uses the canonical runtime engine in `infinite_holdem.py` and
logs one example per real decision point. The output keeps both raw card
strings and count-based encodings so the dataset is easy to inspect and
ready for modeling.
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agents.simple_agent import (
    CallStationAgent,
    EquityMonteCarloAgent,
    RandomAgent,
    TightAggressiveAgent,
)
from infinite_holdem import Card, InfiniteHoldemGame, card_to_index


VARIANT_TAG = "infinite_holdem_v1"


@dataclass
class TrainingExample:
    """Canonical phase-1 training example for the infinite variant."""

    hole_cards: List[str]
    board_cards: List[str]
    hole_card_counts: List[int]
    board_card_counts: List[int]
    pot_size: int
    stack: int
    opponent_stack: int
    position: int
    street: str
    to_call: int
    current_bets: List[int]
    legal_actions: List[str]
    min_raise_to: Optional[int]
    max_raise_to: Optional[int]
    action: str
    raise_amount: Optional[int]
    reward: float
    hand_id: int
    step_in_hand: int
    has_duplicates: bool
    variant_tag: str


def cards_to_count_encoding(cards: List[Card]) -> List[int]:
    """Encode cards as 52-dimensional counts instead of one-hot flags."""

    counts = [0] * 52
    for card in cards:
        counts[card_to_index(card)] += 1
    return counts


def _cards_to_strings(cards: List[Card]) -> List[str]:
    return [str(card) for card in cards]


def _has_duplicates(cards: List[Card]) -> bool:
    card_strings = _cards_to_strings(cards)
    return len(card_strings) != len(set(card_strings))


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
            return ("check", None) if "check" in legal_actions else ("call", None)
        if amount is None:
            amount = min_raise_to
        return "raise", max(min_raise_to, min(amount, max_raise_to))

    if action in legal_actions:
        return action, None

    for fallback in ("check", "call", "fold"):
        if fallback in legal_actions:
            return fallback, None

    return legal_actions[0], None


def _make_agent(label: str, rng: random.Random):
    if label == "random":
        return RandomAgent()
    if label == "call_station":
        return CallStationAgent()
    if label == "tag":
        return TightAggressiveAgent(aggression=rng.uniform(0.45, 0.75))
    if label == "equity":
        return EquityMonteCarloAgent(simulations=40, aggression=rng.uniform(0.35, 0.65))
    raise ValueError(f"Unknown agent label: {label}")


def _select_agents(agent_mix: str, hand_id: int):
    rng = random.Random(hand_id)
    if agent_mix == "random":
        return RandomAgent(), RandomAgent()
    if agent_mix == "tag":
        return _make_agent("tag", rng), _make_agent("tag", rng)
    if agent_mix == "equity":
        return _make_agent("equity", rng), _make_agent("equity", rng)
    if agent_mix == "mixed":
        labels = rng.sample(["random", "call_station", "tag", "equity"], 2)
        return _make_agent(labels[0], rng), _make_agent(labels[1], rng)
    raise ValueError(f"Unsupported agent mix: {agent_mix}")


def simulate_hand(
    agent0,
    agent1,
    hand_id: int,
    starting_stack: int = 10000,
    seed: Optional[int] = None,
) -> List[TrainingExample]:
    """Simulate one full heads-up hand and log every decision point."""

    game = InfiniteHoldemGame(
        num_players=2,
        starting_stack=starting_stack,
        small_blind=50,
        big_blind=100,
        seed=seed,
    )
    agents = [agent0, agent1]
    examples: List[TrainingExample] = []
    step_in_hand = 0

    game.deal_hole_cards()

    while not game.is_hand_over():
        if game.actor_index is None:
            if game.is_betting_round_complete():
                game.advance_street()
                continue
            raise RuntimeError("Engine reached a non-terminal state with no current actor")

        actor = game.actor_index
        legal_actions = game.get_legal_actions(actor)
        example = TrainingExample(
            hole_cards=_cards_to_strings(game.hole_cards[actor]),
            board_cards=_cards_to_strings(game.board),
            hole_card_counts=cards_to_count_encoding(game.hole_cards[actor]),
            board_card_counts=cards_to_count_encoding(game.board),
            pot_size=game.pot,
            stack=game.stacks[actor],
            opponent_stack=game.stacks[1 - actor],
            position=actor,
            street=game.street,
            to_call=game.get_to_call(actor),
            current_bets=list(game.current_bets),
            legal_actions=list(legal_actions),
            min_raise_to=game.get_min_raise_to(actor),
            max_raise_to=game.get_max_raise_to(actor),
            action="",
            raise_amount=None,
            reward=0.0,
            hand_id=hand_id,
            step_in_hand=step_in_hand,
            has_duplicates=False,
            variant_tag=VARIANT_TAG,
        )

        action, raise_amount = agents[actor].get_action(game, actor)
        action, raise_amount = _normalize_action(game, actor, action, raise_amount)
        example.action = action
        example.raise_amount = raise_amount
        examples.append(example)

        game.apply_action(actor, action, raise_amount)
        step_in_hand += 1

    winners, payoffs = game.settle_pot()
    _ = winners

    all_cards = game.hole_cards[0] + game.hole_cards[1] + game.board
    hand_has_duplicates = _has_duplicates(all_cards)
    for example in examples:
        example.reward = payoffs[example.position]
        example.has_duplicates = hand_has_duplicates

    return examples


def generate_dataset(
    num_hands: int = 1000,
    output_file: str = "data/infinite_holdem_training_data.json",
    agent_mix: str = "mixed",
) -> List[Dict]:
    """Generate a JSON dataset of decision-point examples."""

    print(f"Generating {num_hands} hands of Infinite Hold'em...")
    print(f"Agent mix: {agent_mix}")
    print("Using the canonical engine-driven heads-up action loop.")

    all_examples: List[TrainingExample] = []
    hands_with_duplicates = 0
    hand_reward_sums: List[float] = []

    for hand_id in range(num_hands):
        if (hand_id + 1) % 100 == 0:
            print(f"  Completed {hand_id + 1} hands...")

        agent0, agent1 = _select_agents(agent_mix, hand_id)

        try:
            hand_examples = simulate_hand(agent0, agent1, hand_id=hand_id, seed=hand_id)
        except Exception as exc:
            print(f"  Error in hand {hand_id}: {exc}")
            continue

        if not hand_examples:
            continue

        all_examples.extend(hand_examples)
        if hand_examples[0].has_duplicates:
            hands_with_duplicates += 1

        payoffs_by_position = {}
        for example in hand_examples:
            payoffs_by_position[example.position] = example.reward
        if len(payoffs_by_position) == 1:
            only_position, only_reward = next(iter(payoffs_by_position.items()))
            payoffs_by_position[1 - only_position] = -only_reward
        hand_reward_sums.append(sum(payoffs_by_position.values()))

    data = [asdict(example) for example in all_examples]

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)

    print(f"\n{'=' * 50}")
    print("DATASET STATISTICS")
    print("=" * 50)
    print(f"Total training examples: {len(data)}")
    print(f"Hands with duplicates: {hands_with_duplicates}/{num_hands}")
    print(f"Average decisions per hand: {len(data) / num_hands:.2f}")

    actions = [example["action"] for example in data]
    print("\nAction distribution:")
    for action in ("fold", "check", "call", "raise"):
        count = actions.count(action)
        percentage = (100.0 * count / len(actions)) if actions else 0.0
        print(f"  {action}: {count} ({percentage:.1f}%)")

    max_abs_reward_sum = max((abs(value) for value in hand_reward_sums), default=0.0)
    print(f"\nMax absolute hand-level reward sum: {max_abs_reward_sum:.1f}")
    print(f"Saved to {output_file}")

    return data


if __name__ == "__main__":
    dataset = generate_dataset(
        num_hands=200,
        output_file="data/infinite_holdem_training_data.json",
        agent_mix="mixed",
    )

    print(f"\n{'=' * 50}")
    print("SAMPLE TRAINING EXAMPLES")
    print("=" * 50)

    for index, example in enumerate(dataset[:3], start=1):
        print(f"\nExample {index}:")
        print(f"  Street: {example['street']}, Position: P{example['position']}")
        print(f"  Hole cards: {example['hole_cards']}")
        print(f"  Board cards: {example['board_cards']}")
        print(f"  Pot: {example['pot_size']}, To call: {example['to_call']}")
        print(f"  Legal actions: {example['legal_actions']}")
        print(f"  Current bets: {example['current_bets']}")
        print(f"  Action: {example['action']}", end="")
        if example["raise_amount"] is not None:
            print(f" to {example['raise_amount']}", end="")
        print(f"\n  Reward: {example['reward']}")
        print(f"  Has duplicates: {example['has_duplicates']}")
        print(f"  Variant tag: {example['variant_tag']}")
