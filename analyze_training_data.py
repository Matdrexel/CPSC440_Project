"""
analyze_training_data.py - Analyze and visualize hold'em training datasets.

This version supports the newer nested `state_vector` schema used by both the
infinite self-play generator and the IRC conversion pipeline.

Usage:
    python analyze_training_data.py data/infinite_holdem_training_data.json
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Keep matplotlib/font caches inside the project so plots work reliably when
# the default home-directory cache path is not writable.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, ".matplotlib"))
# Use a headless backend so plot saving works from the terminal/venv without a GUI.
os.environ.setdefault("MPLBACKEND", "Agg")

from infinite_holdem import index_to_card

GRAPHS_DIR = os.path.join(PROJECT_ROOT, "graphs")
STREETS = ["preflop", "flop", "turn", "river"]
POSITION_LABELS = {
    0: "BB/OOP",
    1: "BTN/IP",
}
DATASET_PROFILES = {
    "standard_holdem_irc": {
        "slug": "irc_standard_holdem",
        "title": "IRC Standard Hold'em",
        "colors": ["#2a9d8f", "#457b9d", "#8ecae6", "#90be6d"],
    },
    "infinite_holdem_v1": {
        "slug": "infinite_holdem",
        "title": "Infinite Hold'em",
        "colors": ["#e76f51", "#f4a261", "#e9c46a", "#d62828"],
    },
}


try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: install matplotlib for plots (pip install matplotlib)")

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def load_data(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _street_from_one_hot(one_hot: List[int]) -> str:
    if len(one_hot) != 4:
        raise ValueError(f"Street one-hot must have length 4, got {one_hot}")
    try:
        return STREETS[one_hot.index(1)]
    except ValueError as exc:
        raise ValueError(f"Invalid street one-hot vector: {one_hot}") from exc


def _decode_card_index(card_index: int) -> Optional[str]:
    if card_index == -1:
        return None
    return str(index_to_card(card_index))


def _decode_cards(card_indices: List[int], *, drop_undealt: bool = True) -> List[str]:
    decoded: List[str] = []
    for card_index in card_indices:
        card = _decode_card_index(card_index)
        if card is None and drop_undealt:
            continue
        if card is not None:
            decoded.append(card)
    return decoded


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sanitize_slug(value: str) -> str:
    sanitized = []
    for char in value.lower():
        if char.isalnum():
            sanitized.append(char)
        else:
            sanitized.append("_")
    slug = "".join(sanitized).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "dataset"


def _detect_dataset_profile(filepath: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    variant_counts = Counter(example.get("variant_tag", "unknown") for example in data)
    variant_tag = variant_counts.most_common(1)[0][0] if variant_counts else "unknown"

    if variant_tag in DATASET_PROFILES:
        profile = dict(DATASET_PROFILES[variant_tag])
    else:
        stem = os.path.splitext(os.path.basename(filepath))[0]
        profile = {
            "slug": _sanitize_slug(variant_tag if variant_tag != "unknown" else stem),
            "title": stem.replace("_", " ").title(),
            "colors": ["#577590", "#43aa8b", "#90be6d", "#f9c74f"],
        }

    profile["variant_tag"] = variant_tag
    profile["source_stem"] = os.path.splitext(os.path.basename(filepath))[0]
    profile["graphs_dir"] = os.path.join(GRAPHS_DIR, profile["slug"])
    return profile


def _normalize_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Project stored JSON rows into a stable analysis shape."""

    if "state_vector" in example:
        state = example["state_vector"]
        position = state["position"]
        my_bet = state["my_bet_this_street"]
        opp_bet = state["opp_bet_this_street"]
        return {
            "hand_id": example["hand_id"],
            "step_in_hand": example["step_in_hand"],
            "action": example["action"],
            "raise_amount": example.get("raise_amount"),
            "reward": example["reward"],
            "has_duplicates": example.get("has_duplicates", False),
            "variant_tag": example.get("variant_tag", "unknown"),
            "state_vector": state,
            "position": position,
            "street": _street_from_one_hot(state["street_one_hot"]),
            "hole_cards": _decode_cards(state["hole_cards"]),
            "hole_card_indices": list(state["hole_cards"]),
            "board_cards": _decode_cards(state["board_cards"]),
            "board_card_indices": list(state["board_cards"]),
            "pot_size": state["pot"],
            "my_stack": state["my_stack"],
            "opponent_stack": state["opponent_stack"],
            "my_bet_this_street": my_bet,
            "opp_bet_this_street": opp_bet,
            "to_call": max(0, opp_bet - my_bet),
        }

    # Legacy fallback for older saved data.
    return {
        **example,
        "hole_card_indices": example.get("hole_cards", []),
        "board_card_indices": example.get("board_cards", []),
        "my_stack": example.get("stack"),
        "my_bet_this_street": None,
        "opp_bet_this_street": None,
        "state_vector": None,
    }


def normalize_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_normalize_example(example) for example in data]


@dataclass
class HandSummary:
    hand_id: int
    players: Dict[int, Dict[str, Any]]
    board: List[str]
    actions: List[Dict[str, Any]]
    has_duplicates: bool
    winner: Optional[int]


def reconstruct_hands(data: List[Dict[str, Any]]) -> Dict[int, HandSummary]:
    hands: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {
            "actions": [],
            "players": {},
            "board": [],
            "has_duplicates": False,
        }
    )

    for example in data:
        hand_id = example["hand_id"]
        position = example["position"]

        if position not in hands[hand_id]["players"]:
            hands[hand_id]["players"][position] = {
                "hole_cards": example["hole_cards"],
                "reward": example["reward"],
            }

        hands[hand_id]["actions"].append(
            {
                "step": example["step_in_hand"],
                "position": position,
                "street": example["street"],
                "action": example["action"],
                "raise_amount": example.get("raise_amount"),
                "pot": example["pot_size"],
                "to_call": example["to_call"],
            }
        )

        if len(example["board_cards"]) > len(hands[hand_id]["board"]):
            hands[hand_id]["board"] = example["board_cards"]

        hands[hand_id]["has_duplicates"] = hands[hand_id]["has_duplicates"] or example.get(
            "has_duplicates", False
        )

    summaries: Dict[int, HandSummary] = {}
    for hand_id, hand in hands.items():
        winner = None
        positive_positions = [pos for pos, player in hand["players"].items() if player["reward"] > 0]
        if len(positive_positions) == 1:
            winner = positive_positions[0]

        summaries[hand_id] = HandSummary(
            hand_id=hand_id,
            players=hand["players"],
            board=hand["board"],
            actions=sorted(hand["actions"], key=lambda action: action["step"]),
            has_duplicates=hand["has_duplicates"],
            winner=winner,
        )

    return summaries


def _position_label(position: int) -> str:
    return POSITION_LABELS.get(position, f"Pos {position}")


def _hand_reward_sum(hand: HandSummary) -> float:
    return sum(player["reward"] for player in hand.players.values())


def print_summary_stats(data: List[Dict[str, Any]]) -> None:
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    num_examples = len(data)
    hands = reconstruct_hands(data)
    num_hands = len(hands)

    print(f"\nTotal decision examples: {num_examples}")
    print(f"Total hands played: {num_hands}")
    print(f"Average decisions per hand: {num_examples / num_hands:.1f}")

    dup_hands = [hand for hand in hands.values() if hand.has_duplicates]
    print(
        f"\nHands with duplicate cards: {len(dup_hands)}/{num_hands} "
        f"({100 * len(dup_hands) / num_hands:.1f}%)"
    )

    actions = [example["action"] for example in data]
    action_counts = Counter(actions)
    print("\nOverall action distribution:")
    for action in ["fold", "check", "call", "raise"]:
        count = action_counts.get(action, 0)
        pct = 100 * count / len(actions) if actions else 0.0
        print(f"  {action}: {count} ({pct:.1f}%)")

    rewards = [example["reward"] for example in data]
    print("\nReward statistics (decision rows):")
    print(f"  Min: {min(rewards)}")
    print(f"  Max: {max(rewards)}")
    print(f"  Mean: {sum(rewards) / len(rewards):.2f}")

    non_zero_hands = sum(1 for hand in hands.values() if abs(_hand_reward_sum(hand)) > 0.01)
    print("\nZero-sum validation:")
    print(f"  Hands with non-zero reward sum: {non_zero_hands}/{num_hands}")
    if non_zero_hands == 0:
        print("  ✓ All hands are zero-sum at the hand level")


def print_hand_replay(hand: HandSummary, verbose: bool = True) -> None:
    print(f"\n{'=' * 60}")
    print(f"HAND #{hand.hand_id}" + (" [HAS DUPLICATES]" if hand.has_duplicates else ""))
    print("=" * 60)

    for position, player in sorted(hand.players.items()):
        cards = " ".join(player["hole_cards"])
        reward = player["reward"]
        win_marker = " 🏆" if reward > 0 else ""
        print(f"{_position_label(position)}: [{cards}] → {reward:+d}{win_marker}")

    if verbose:
        board = " ".join(hand.board) if hand.board else "(no showdown)"
        print(f"\nBoard: {board}")
        current_street = None
        for action in hand.actions:
            if action["street"] != current_street:
                current_street = action["street"]
                print(f"\n--- {current_street.upper()} ---")

            action_label = action["action"]
            if action["raise_amount"] is not None:
                action_label += f" {action['raise_amount']}"

            print(
                f"  {_position_label(action['position'])}: {action_label:12} "
                f"(pot: {action['pot']}, to_call: {action['to_call']})"
            )


def replay_hands(data: List[Dict[str, Any]], hand_ids: Optional[List[int]] = None, max_hands: int = 5) -> None:
    hands = reconstruct_hands(data)

    if hand_ids is None:
        hand_ids = sorted(hands.keys())[:max_hands]

    print(f"\nReplaying {len(hand_ids)} hands...")
    for hand_id in hand_ids:
        if hand_id in hands:
            print_hand_replay(hands[hand_id])


def replay_hands_with_duplicates(data: List[Dict[str, Any]], max_hands: int = 5) -> None:
    hands = reconstruct_hands(data)
    dup_hands = [hand for hand in hands.values() if hand.has_duplicates]

    print(f"\n{'=' * 60}")
    print(f"HANDS WITH DUPLICATE CARDS ({len(dup_hands)} total)")
    print("=" * 60)

    for hand in dup_hands[:max_hands]:
        print_hand_replay(hand)


def analyze_agent_tendencies(data: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print("AGENT TENDENCIES (POKER STATS)")
    print("=" * 60)

    hands = reconstruct_hands(data)
    num_hands = len(hands)

    voluntarily_played = 0
    preflop_raises = 0
    postflop_calls = 0
    postflop_raises = 0

    for hand in hands.values():
        preflop_actions = [action for action in hand.actions if action["street"] == "preflop"]
        postflop_actions = [action for action in hand.actions if action["street"] in ["flop", "turn", "river"]]

        if any(action["action"] in ["call", "raise"] for action in preflop_actions):
            voluntarily_played += 1
        if any(action["action"] == "raise" for action in preflop_actions):
            preflop_raises += 1

        postflop_calls += sum(1 for action in postflop_actions if action["action"] == "call")
        postflop_raises += sum(1 for action in postflop_actions if action["action"] == "raise")

    vpip_pct = (voluntarily_played / num_hands) * 100 if num_hands else 0.0
    pfr_pct = (preflop_raises / num_hands) * 100 if num_hands else 0.0
    aggression_factor = postflop_raises / max(1, postflop_calls)

    print(f"Overall VPIP (Voluntarily Put in Pot): {vpip_pct:.1f}%")
    print(f"Overall PFR (Preflop Raise):           {pfr_pct:.1f}%")
    print(f"Postflop Aggression Factor (AF):       {aggression_factor:.2f}")


def validate_decision_logic(data: List[Dict[str, Any]], manual_review_count: int = 3) -> None:
    print("\n" + "=" * 60)
    print("LOGICAL SANITY CHECKS")
    print("=" * 60)

    illogical_folds = 0
    massive_overbets = 0

    for example in data:
        if example["action"] == "fold" and example["to_call"] == 0:
            illogical_folds += 1

        if (
            example["action"] == "raise"
            and example["raise_amount"] is not None
            and example["pot_size"] > 0
            and example["raise_amount"] > example["pot_size"] * 10
        ):
            massive_overbets += 1

    print(f"Illogical folds (folding when free to check): {illogical_folds}")
    print(f"Massive overbets (>10x pot):                  {massive_overbets}")

    if illogical_folds == 0 and massive_overbets == 0:
        print("  ✓ Automated sanity checks passed clean.")
    else:
        print("  ! Warning: found possible anomalies in decision logic.")

    hands = reconstruct_hands(data)
    sorted_hands = sorted(
        hands.values(),
        key=lambda hand: max((action["pot"] for action in hand.actions), default=0),
        reverse=True,
    )
    subset = sorted_hands[: max(10, len(sorted_hands) // 10)] if sorted_hands else []
    if not subset:
        return

    print(f"\nSampling {min(manual_review_count, len(subset))} large-pot hands for manual review...")
    for hand in random.sample(subset, min(manual_review_count, len(subset))):
        print_hand_replay(hand, verbose=True)


def analyze_actions_by_street(data: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print("ACTION DISTRIBUTION BY STREET")
    print("=" * 60)

    street_actions: Dict[str, List[str]] = defaultdict(list)
    for example in data:
        street_actions[example["street"]].append(example["action"])

    for street in STREETS:
        if street not in street_actions:
            continue
        actions = street_actions[street]
        counts = Counter(actions)
        total = len(actions)
        print(f"\n{street.upper()} ({total} decisions):")
        for action in ["fold", "check", "call", "raise"]:
            count = counts.get(action, 0)
            pct = 100 * count / total if total else 0.0
            bar = "█" * int(pct / 2)
            print(f"  {action:6}: {count:4} ({pct:5.1f}%) {bar}")


def analyze_actions_by_position(data: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print("ACTION DISTRIBUTION BY POSITION")
    print("=" * 60)

    pos_actions: Dict[int, List[str]] = defaultdict(list)
    for example in data:
        pos_actions[example["position"]].append(example["action"])

    for position in [0, 1]:
        actions = pos_actions[position]
        counts = Counter(actions)
        total = len(actions)
        print(f"\n{_position_label(position)} ({total} decisions):")
        for action in ["fold", "check", "call", "raise"]:
            count = counts.get(action, 0)
            pct = 100 * count / total if total else 0.0
            bar = "█" * int(pct / 2)
            print(f"  {action:6}: {count:4} ({pct:5.1f}%) {bar}")


def analyze_raise_sizes(data: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print("RAISE SIZE ANALYSIS")
    print("=" * 60)

    raises = [example for example in data if example["action"] == "raise" and example.get("raise_amount")]
    if not raises:
        print("No raises found in data.")
        return

    street_raises: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for example in raises:
        pot = example["pot_size"]
        raise_amt = example["raise_amount"]
        pot_pct = (raise_amt / pot * 100) if pot > 0 else 0.0
        street_raises[example["street"]].append(
            {
                "amount": raise_amt,
                "pot_pct": pot_pct,
            }
        )

    for street in STREETS:
        if street not in street_raises:
            continue
        entries = street_raises[street]
        amounts = [entry["amount"] for entry in entries]
        pot_pcts = [entry["pot_pct"] for entry in entries]
        print(f"\n{street.upper()} ({len(entries)} raises):")
        print(
            f"  Raise amount: min={min(amounts)}, max={max(amounts)}, "
            f"avg={_mean(amounts):.0f}"
        )
        print(
            f"  As % of pot:  min={min(pot_pcts):.0f}%, max={max(pot_pcts):.0f}%, "
            f"avg={_mean(pot_pcts):.0f}%"
        )


def analyze_rewards(data: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print("REWARD ANALYSIS")
    print("=" * 60)

    hands = reconstruct_hands(data)
    winners: List[float] = []
    losers: List[float] = []

    for hand in hands.values():
        for player in hand.players.values():
            if player["reward"] > 0:
                winners.append(player["reward"])
            elif player["reward"] < 0:
                losers.append(player["reward"])

    print(f"\nWinning hands: {len(winners)}")
    if winners:
        print(f"  Average win: +{_mean(winners):.0f}")
        print(f"  Biggest win: +{max(winners)}")

    print(f"\nLosing hands: {len(losers)}")
    if losers:
        print(f"  Average loss: {_mean(losers):.0f}")
        print(f"  Biggest loss: {min(losers)}")

    print("\nAverage reward by action:")
    action_rewards: Dict[str, List[float]] = defaultdict(list)
    for example in data:
        action_rewards[example["action"]].append(example["reward"])

    for action in ["fold", "check", "call", "raise"]:
        rewards = action_rewards.get(action, [])
        if rewards:
            print(f"  {action}: {_mean(rewards):+.1f}")


def analyze_duplicates(data: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print("DUPLICATE CARD ANALYSIS")
    print("=" * 60)

    hands = reconstruct_hands(data)
    dup_hands = [hand for hand in hands.values() if hand.has_duplicates]
    normal_hands = [hand for hand in hands.values() if not hand.has_duplicates]

    print(f"\nHands with duplicates: {len(dup_hands)}")
    print(f"Hands without duplicates: {len(normal_hands)}")

    if not dup_hands:
        print("\nNo duplicate hands to analyze.")
        return

    def get_win_amounts(hand_list: List[HandSummary]) -> List[float]:
        amounts: List[float] = []
        for hand in hand_list:
            for player in hand.players.values():
                if player["reward"] > 0:
                    amounts.append(player["reward"])
        return amounts

    dup_wins = get_win_amounts(dup_hands)
    normal_wins = get_win_amounts(normal_hands)

    print("\nAverage winning amount:")
    if dup_wins:
        print(f"  Hands with duplicates: +{_mean(dup_wins):.0f}")
    if normal_wins:
        print(f"  Normal hands: +{_mean(normal_wins):.0f}")

    print("\nExample hands with duplicates:")
    for hand in dup_hands[:3]:
        cards_seen: List[str] = []
        for player in hand.players.values():
            cards_seen.extend(player["hole_cards"])
        cards_seen.extend(hand.board)

        duplicates = [card for card, count in Counter(cards_seen).items() if count > 1]
        print(f"\n  Hand #{hand.hand_id}:")
        print(f"    Duplicate cards: {duplicates}")
        for position, player in hand.players.items():
            print(f"    {_position_label(position)}: {player['hole_cards']} → {player['reward']:+d}")
        print(f"    Board: {hand.board}")


def _vector_from_state_vector(state_vector: Dict[str, Any]) -> List[float]:
    vector = []
    vector.extend(float(value) for value in state_vector["hole_cards"])
    vector.extend(float(value) for value in state_vector["board_cards"])
    vector.extend(float(value) for value in state_vector["street_one_hot"])
    vector.append(float(state_vector["position"]))
    vector.append(float(state_vector["my_stack"]))
    vector.append(float(state_vector["opponent_stack"]))
    vector.append(float(state_vector["pot"]))
    vector.append(float(state_vector["my_bet_this_street"]))
    vector.append(float(state_vector["opp_bet_this_street"]))
    return vector


def extract_ml_features(data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not HAS_NUMPY:
        print("NumPy required for ML feature extraction")
        return None

    print("\n" + "=" * 60)
    print("ML FEATURE EXTRACTION")
    print("=" * 60)

    features: List[List[float]] = []
    actions: List[int] = []
    rewards: List[float] = []
    action_to_idx = {"fold": 0, "check": 1, "call": 2, "raise": 3}

    for example in data:
        state_vector = example.get("state_vector")
        if state_vector is None:
            continue
        features.append(_vector_from_state_vector(state_vector))
        actions.append(action_to_idx.get(example["action"], -1))
        rewards.append(example["reward"])

    if not features:
        print("No `state_vector` entries found to export.")
        return None

    feature_names = [
        "hole_card_0",
        "hole_card_1",
        "board_card_0",
        "board_card_1",
        "board_card_2",
        "board_card_3",
        "board_card_4",
        "street_preflop",
        "street_flop",
        "street_turn",
        "street_river",
        "position",
        "my_stack",
        "opponent_stack",
        "pot",
        "my_bet_this_street",
        "opp_bet_this_street",
    ]

    X = np.array(features, dtype=float)
    y_action = np.array(actions, dtype=int)
    y_reward = np.array(rewards, dtype=float)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Actions shape: {y_action.shape}")
    print(f"Rewards shape: {y_reward.shape}")
    print(f"Vector length: {X.shape[1]}")

    return {
        "X": X,
        "y_action": y_action,
        "y_reward": y_reward,
        "feature_names": feature_names,
        "action_map": {value: key for key, value in action_to_idx.items()},
    }


def plot_action_distribution(
    data: List[Dict[str, Any]],
    dataset_profile: Dict[str, Any],
    save_path: Optional[str] = None,
) -> None:
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return

    actions = [example["action"] for example in data]
    counts = Counter(actions)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = dataset_profile["colors"]
    dataset_title = dataset_profile["title"]

    axes[0].pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors[: len(labels)], startangle=90)
    axes[0].set_title(f"{dataset_title}: Overall Action Distribution")

    street_actions: Dict[str, Counter] = defaultdict(Counter)
    for example in data:
        street_actions[example["street"]][example["action"]] += 1

    x = range(len(STREETS))
    width = 0.2
    for index, action in enumerate(["fold", "check", "call", "raise"]):
        values = [street_actions[street].get(action, 0) for street in STREETS]
        axes[1].bar([value + index * width for value in x], values, width, label=action, color=colors[index])

    axes[1].set_xlabel("Street")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{dataset_title}: Actions by Street")
    axes[1].set_xticks([value + 1.5 * width for value in x])
    axes[1].set_xticklabels(STREETS)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_reward_distribution(
    data: List[Dict[str, Any]],
    dataset_profile: Dict[str, Any],
    save_path: Optional[str] = None,
) -> None:
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return

    rewards = [example["reward"] for example in data]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = dataset_profile["colors"]
    dataset_title = dataset_profile["title"]

    axes[0].hist(rewards, bins=50, color=colors[1], edgecolor="black", alpha=0.7)
    axes[0].axvline(x=0, color="red", linestyle="--", label="Break even")
    axes[0].set_xlabel("Reward")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"{dataset_title}: Reward Distribution")
    axes[0].legend()

    action_rewards: Dict[str, List[float]] = defaultdict(list)
    for example in data:
        action_rewards[example["action"]].append(example["reward"])

    actions = ["fold", "check", "call", "raise"]
    avg_rewards = [_mean(action_rewards[action]) for action in actions]
    bar_colors = [colors[0] if reward < 0 else colors[1] for reward in avg_rewards]

    axes[1].bar(actions, avg_rewards, color=bar_colors, edgecolor="black")
    axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    axes[1].set_xlabel("Action")
    axes[1].set_ylabel("Average Reward")
    axes[1].set_title(f"{dataset_title}: Average Reward by Action")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_pot_sizes(
    data: List[Dict[str, Any]],
    dataset_profile: Dict[str, Any],
    save_path: Optional[str] = None,
) -> None:
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return

    street_pots: Dict[str, List[float]] = defaultdict(list)
    for example in data:
        street_pots[example["street"]].append(example["pot_size"])

    fig, ax = plt.subplots(figsize=(10, 6))
    box_data = [street_pots[street] for street in STREETS if street_pots[street]]
    labels = [street for street in STREETS if street_pots[street]]

    bp = ax.boxplot(box_data, widths=0.6, patch_artist=True)
    colors = dataset_profile["colors"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels)
    ax.set_xlabel("Street")
    ax.set_ylabel("Pot Size")
    ax.set_title(f"{dataset_profile['title']}: Pot Size Distribution by Street")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main(filepath: str):
    print(f"\nLoading data from: {filepath}")
    raw_data = load_data(filepath)
    data = normalize_data(raw_data)
    print(f"Loaded {len(data)} decision examples")
    dataset_profile = _detect_dataset_profile(filepath, data)
    graphs_dir = dataset_profile["graphs_dir"]

    os.makedirs(graphs_dir, exist_ok=True)
    print(f"Detected dataset: {dataset_profile['title']} ({dataset_profile['variant_tag']})")

    print_summary_stats(data)
    analyze_agent_tendencies(data)
    analyze_actions_by_street(data)
    analyze_actions_by_position(data)
    analyze_raise_sizes(data)
    analyze_rewards(data)
    analyze_duplicates(data)
    validate_decision_logic(data, manual_review_count=3)
    ml_data = extract_ml_features(data)

    if HAS_MATPLOTLIB:
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print(f"Saving to directory: {graphs_dir}")
        print("=" * 60)
        plot_action_distribution(
            data,
            dataset_profile,
            os.path.join(graphs_dir, f"{dataset_profile['slug']}_action_distribution.png"),
        )
        plot_reward_distribution(
            data,
            dataset_profile,
            os.path.join(graphs_dir, f"{dataset_profile['slug']}_reward_distribution.png"),
        )
        plot_pot_sizes(
            data,
            dataset_profile,
            os.path.join(graphs_dir, f"{dataset_profile['slug']}_pot_sizes.png"),
        )
        print("\nPlots saved successfully!")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    return data, ml_data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_training_data.py <path_to_json>")
        print("\nExample:")
        print("  python analyze_training_data.py data/infinite_holdem_training_data.json")
        sys.exit(1)

    filepath = sys.argv[1]
    main(filepath)
