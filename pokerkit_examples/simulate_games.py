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

from agents.realistic_agents import (
    NitAgent, TightAggressiveAgent, LooseAggressiveAgent,
    LoosePassiveAgent, BalancedAgent, RandomAgent
)
from infinite_holdem import Card, InfiniteHoldemGame, card_to_index


VARIANT_TAG = "infinite_holdem_v1"


@dataclass
class StateVector:
    """Serializable state payload that maps directly to the later model vector."""

    hole_cards: List[int]
    board_cards: List[int]
    street_one_hot: List[int]
    position: int
    my_stack: int
    opponent_stack: int
    pot: int
    my_bet_this_street: int
    opp_bet_this_street: int


@dataclass
class TrainingExample:
    """Training example containing the exact requested JSON schema."""

    state_vector: StateVector
    action: str
    raise_amount: Optional[int]
    reward: float
    hand_id: int
    step_in_hand: int
    has_duplicates: bool
    variant_tag: str


def _has_duplicates(cards: List[Card]) -> bool:
    card_strings = [str(card) for card in cards]
    return len(card_strings) != len(set(card_strings))


def _encode_hole_cards(cards: List[Card]) -> List[int]:
    return [card_to_index(card) for card in cards]


def _encode_board_cards(cards: List[Card]) -> List[int]:
    encoded = [card_to_index(card) for card in cards]
    while len(encoded) < 5:
        encoded.append(-1)
    return encoded[:5]


def _street_one_hot(street: str) -> List[int]:
    mapping = {
        "preflop": [1, 0, 0, 0],
        "flop": [0, 1, 0, 0],
        "turn": [0, 0, 1, 0],
        "river": [0, 0, 0, 1],
    }
    if street not in mapping:
        raise ValueError(f"Unsupported street for state serialization: {street}")
    return mapping[street]


def _serialize_position(player_idx: int) -> int:
    """Map engine player index to requested semantic position."""

    return 1 if player_idx == 0 else 0


def _build_state_vector(game: InfiniteHoldemGame, actor: int) -> StateVector:
    return StateVector(
        hole_cards=_encode_hole_cards(game.hole_cards[actor]),
        board_cards=_encode_board_cards(game.board),
        street_one_hot=_street_one_hot(game.street),
        position=_serialize_position(actor),
        my_stack=game.stacks[actor],
        opponent_stack=game.stacks[1 - actor],
        pot=game.pot,
        my_bet_this_street=game.current_bets[actor],
        opp_bet_this_street=game.current_bets[1 - actor],
    )


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
    if label == "nit":
        return NitAgent()
    if label == "tight_aggressive":
        return TightAggressiveAgent()
    if label == "loose_aggressive":
        return LooseAggressiveAgent()
    if label == "loose_passive":
        return LoosePassiveAgent()
    if label == "balanced":
        return BalancedAgent()
    raise ValueError(f"Unknown agent label: {label}")


def _select_agents(agent_mix: str, hand_id: int):
    rng = random.Random(hand_id)

    if agent_mix == "random":
        return _make_agent("random", rng), _make_agent("random", rng)
    if agent_mix == "nit":
        return _make_agent("nit", rng), _make_agent("nit", rng)
    if agent_mix == "tight_aggressive":
        return _make_agent("tight_aggressive", rng), _make_agent("tight_aggressive", rng)
    if agent_mix == "loose_aggressive":
        return _make_agent("loose_aggressive", rng), _make_agent("loose_aggressive", rng)
    if agent_mix == "loose_passive":
        return _make_agent("loose_passive", rng), _make_agent("loose_passive", rng)
    if agent_mix == "balanced":
        return _make_agent("balanced", rng), _make_agent("balanced", rng)
    if agent_mix == "mixed":
        labels = rng.sample(
            [
                "random",
                "nit",
                "tight_aggressive",
                "loose_aggressive",
                "loose_passive",
                "balanced",
            ],
            2,
        )
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
    example_owners: List[int] = []
    step_in_hand = 0

    game.deal_hole_cards()

    while not game.is_hand_over():
        if game.actor_index is None:
            if game.is_betting_round_complete():
                game.advance_street()
                continue
            raise RuntimeError("Engine reached a non-terminal state with no current actor")

        actor = game.actor_index
        example = TrainingExample(
            state_vector=_build_state_vector(game, actor),
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
        example_owners.append(actor)

        game.apply_action(actor, action, raise_amount)
        step_in_hand += 1

    winners, payoffs = game.settle_pot()
    _ = winners

    all_cards = game.hole_cards[0] + game.hole_cards[1] + game.board
    hand_has_duplicates = _has_duplicates(all_cards)
    for example, owner in zip(examples, example_owners):
        example.reward = payoffs[owner]
        example.has_duplicates = hand_has_duplicates

    return examples


def generate_dataset(
    num_hands: int = 10000,
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
            payoffs_by_position[example.state_vector.position] = example.reward
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
        num_hands=100000,
        output_file="data/infinite_holdem_training_data.json",
        agent_mix="mixed",
    )

    print(f"\n{'=' * 50}")
    print("SAMPLE TRAINING EXAMPLES")
    print("=" * 50)

    for index, example in enumerate(dataset[:3], start=1):
        state = example["state_vector"]
        print(f"\nExample {index}:")
        print(f"  Hole cards: {state['hole_cards']}")
        print(f"  Board cards: {state['board_cards']}")
        print(f"  Street one-hot: {state['street_one_hot']}")
        print(f"  Position: {state['position']}")
        print(f"  My stack: {state['my_stack']}")
        print(f"  Opponent stack: {state['opponent_stack']}")
        print(f"  Pot: {state['pot']}")
        print(f"  My bet this street: {state['my_bet_this_street']}")
        print(f"  Opp bet this street: {state['opp_bet_this_street']}")
        print(f"  Action: {example['action']}", end="")
        if example["raise_amount"] is not None:
            print(f" to {example['raise_amount']}", end="")
        print(f"\n  Reward: {example['reward']}")
        print(f"  Has duplicates: {example['has_duplicates']}")
        print(f"  Variant tag: {example['variant_tag']}")
