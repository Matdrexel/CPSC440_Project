#!/usr/bin/env python3
"""
Reevaluate standard hold'em trajectory data under infinite-hold'em rankings.

This script is intended to run after `convert_irc_script.py`. It reads a
 standard-rules JSON trajectory dataset, reevaluates showdown hands under both
 standard and infinite rankings, and writes:

- <prefix>.augmented.json   : original rows plus infinite-outcome metadata
- <prefix>.per_hand.jsonl   : one audit record per hand
- <prefix>.summary.json     : summary metrics and flip breakdowns
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infinite_holdem import Card, InfiniteHandEvaluator, RANK_NAMES, RANK_VALUES, RANKS, SUITS


STANDARD_AUGMENTED_SUFFIX = "augmented_infinite"
STANDARD_IRC_VARIANT = "standard_holdem_irc"
CHANGING_HAND_TYPES = {"STRAIGHT", "FLUSH", "FULL_HOUSE"}


def clean_number(value: float) -> float | int:
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return value


class StandardHandRank(IntEnum):
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9


@dataclass
class StandardHandResult:
    rank: StandardHandRank
    description: str
    cards: List[Card]
    kickers: List[int]

    def _cmp_key(self) -> Tuple[int, Tuple[int, ...]]:
        return self.rank.value, tuple(self.kickers)

    def __lt__(self, other: "StandardHandResult") -> bool:
        return self._cmp_key() < other._cmp_key()

    def __le__(self, other: "StandardHandResult") -> bool:
        return self._cmp_key() <= other._cmp_key()

    def __gt__(self, other: "StandardHandResult") -> bool:
        return self._cmp_key() > other._cmp_key()

    def __ge__(self, other: "StandardHandResult") -> bool:
        return self._cmp_key() >= other._cmp_key()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StandardHandResult):
            return NotImplemented
        return self._cmp_key() == other._cmp_key()


class StandardHandEvaluator:
    """Evaluate classical 5-of-7 no-limit hold'em hands."""

    @staticmethod
    def evaluate(cards: List[Card]) -> StandardHandResult:
        if len(cards) < 5:
            raise ValueError(f"Need at least 5 cards, got {len(cards)}")

        best_hand: Optional[StandardHandResult] = None
        for five_cards in combinations(cards, 5):
            hand = StandardHandEvaluator._evaluate_five(list(five_cards))
            if best_hand is None or hand > best_hand:
                best_hand = hand
        return best_hand

    @staticmethod
    def _evaluate_five(cards: List[Card]) -> StandardHandResult:
        rank_counts = Counter(card.rank for card in cards)
        suit_counts = Counter(card.suit for card in cards)
        rank_values = sorted((card.rank_value for card in cards), reverse=True)
        is_flush = max(suit_counts.values()) == 5
        is_straight = StandardHandEvaluator._is_straight(rank_values)
        max_rank_count = max(rank_counts.values())
        rank_count_values = sorted(rank_counts.values(), reverse=True)

        if is_flush and is_straight:
            high = max(rank_values) if not StandardHandEvaluator._is_wheel(rank_values) else 3
            return StandardHandResult(
                StandardHandRank.STRAIGHT_FLUSH,
                f"Straight Flush, {RANKS[high]} high",
                cards,
                [high],
            )

        if max_rank_count == 4:
            quad_rank = next(rank for rank, count in rank_counts.items() if count == 4)
            kicker = next(rank for rank, count in rank_counts.items() if count != 4)
            return StandardHandResult(
                StandardHandRank.FOUR_OF_A_KIND,
                f"Four of a Kind, {RANK_NAMES[quad_rank]}",
                cards,
                [RANK_VALUES[quad_rank], RANK_VALUES[kicker]],
            )

        if rank_count_values == [3, 2]:
            trips_rank = next(rank for rank, count in rank_counts.items() if count == 3)
            pair_rank = next(rank for rank, count in rank_counts.items() if count == 2)
            return StandardHandResult(
                StandardHandRank.FULL_HOUSE,
                f"Full House, {RANK_NAMES[trips_rank]} full of {RANK_NAMES[pair_rank]}",
                cards,
                [RANK_VALUES[trips_rank], RANK_VALUES[pair_rank]],
            )

        if is_flush:
            return StandardHandResult(
                StandardHandRank.FLUSH,
                f"Flush, {RANKS[rank_values[0]]} high",
                cards,
                rank_values,
            )

        if is_straight:
            high = max(rank_values) if not StandardHandEvaluator._is_wheel(rank_values) else 3
            return StandardHandResult(
                StandardHandRank.STRAIGHT,
                f"Straight, {RANKS[high]} high",
                cards,
                [high],
            )

        if max_rank_count == 3:
            trips_rank = next(rank for rank, count in rank_counts.items() if count == 3)
            kickers = sorted(
                (RANK_VALUES[rank] for rank, count in rank_counts.items() if count == 1),
                reverse=True,
            )
            return StandardHandResult(
                StandardHandRank.THREE_OF_A_KIND,
                f"Three of a Kind, {RANK_NAMES[trips_rank]}",
                cards,
                [RANK_VALUES[trips_rank]] + kickers,
            )

        if rank_count_values == [2, 2, 1]:
            pairs = sorted(
                (RANK_VALUES[rank] for rank, count in rank_counts.items() if count == 2),
                reverse=True,
            )
            kicker = next(RANK_VALUES[rank] for rank, count in rank_counts.items() if count == 1)
            return StandardHandResult(
                StandardHandRank.TWO_PAIR,
                f"Two Pair, {RANK_NAMES[RANKS[pairs[0]]]} and {RANK_NAMES[RANKS[pairs[1]]]}",
                cards,
                pairs + [kicker],
            )

        if max_rank_count == 2:
            pair_rank = next(rank for rank, count in rank_counts.items() if count == 2)
            kickers = sorted(
                (RANK_VALUES[rank] for rank, count in rank_counts.items() if count == 1),
                reverse=True,
            )
            return StandardHandResult(
                StandardHandRank.PAIR,
                f"Pair of {RANK_NAMES[pair_rank]}",
                cards,
                [RANK_VALUES[pair_rank]] + kickers,
            )

        return StandardHandResult(
            StandardHandRank.HIGH_CARD,
            f"High Card, {RANKS[rank_values[0]]}",
            cards,
            rank_values,
        )

    @staticmethod
    def _is_straight(rank_values: List[int]) -> bool:
        unique = sorted(set(rank_values))
        if len(unique) != 5:
            return False
        if unique[-1] - unique[0] == 4:
            return True
        return unique == [0, 1, 2, 3, 12]

    @staticmethod
    def _is_wheel(rank_values: List[int]) -> bool:
        return sorted(set(rank_values)) == [0, 1, 2, 3, 12]


def idx_to_card(index: int) -> Card:
    if not 0 <= index <= 51:
        raise ValueError(f"Card index {index} out of range 0..51")
    return Card.from_string(RANKS[index // 4] + SUITS[index % 4])


@dataclass
class PlayerRollup:
    position: int
    hole_cards: List[int]
    folded: bool = False
    standard_reward: Optional[float] = None


@dataclass
class HandAuditReport:
    hand_id: int
    showdown: bool = False
    showdown_reevaluated: bool = False
    winner_flipped: bool = False
    standard_winners: List[int] = field(default_factory=list)
    infinite_winners: List[int] = field(default_factory=list)
    reward_inferred_standard_winners: List[int] = field(default_factory=list)
    standard_winner_mismatch: bool = False
    board: List[int] = field(default_factory=list)
    standard_hand_types: Dict[int, str] = field(default_factory=dict)
    infinite_hand_types: Dict[int, str] = field(default_factory=dict)
    standard_descriptions: Dict[int, str] = field(default_factory=dict)
    infinite_descriptions: Dict[int, str] = field(default_factory=dict)
    best_hand_changed: Dict[int, bool] = field(default_factory=dict)
    standard_rewards: Dict[int, float | int] = field(default_factory=dict)
    infinite_rewards: Dict[int, float | int] = field(default_factory=dict)
    standard_matchup: str = ""
    infinite_matchup: str = ""
    skip_reason: Optional[str] = None


def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def group_records_by_hand(
    records: List[Dict[str, Any]],
    max_hands: Optional[int] = None,
) -> Tuple[List[int], Dict[int, List[Dict[str, Any]]]]:
    by_hand: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    hand_order: List[int] = []
    seen: set[int] = set()

    for record in records:
        hand_id = int(record["hand_id"])
        if hand_id not in seen:
            seen.add(hand_id)
            hand_order.append(hand_id)
        by_hand[hand_id].append(record)

    if max_hands is not None:
        selected = hand_order[:max_hands]
        return selected, {hand_id: by_hand[hand_id] for hand_id in selected}

    return hand_order, dict(by_hand)


def reconstruct_hand(records: List[Dict[str, Any]]) -> Tuple[Dict[int, PlayerRollup], List[int], float]:
    players: Dict[int, PlayerRollup] = {}
    final_board: List[int] = []
    final_pot = 0.0

    for record in records:
        state = record["state_vector"]
        position = int(state["position"])
        hole_cards = list(state["hole_cards"])
        board_cards = [card for card in state["board_cards"] if card != -1]
        pot_size = float(state["pot"])
        reward = float(record.get("reward", 0.0))

        if position not in players:
            players[position] = PlayerRollup(
                position=position,
                hole_cards=hole_cards,
                standard_reward=reward,
            )
        else:
            player = players[position]
            if player.hole_cards == [-1, -1] and hole_cards != [-1, -1]:
                player.hole_cards = hole_cards

        player = players[position]
        if record.get("action") == "fold":
            player.folded = True
        if player.standard_reward is None:
            player.standard_reward = reward

        if len(board_cards) > len(final_board):
            final_board = board_cards
        final_pot = max(final_pot, pot_size)

    return players, final_board, final_pot


def _winners_from_results(results: Dict[int, Any]) -> List[int]:
    if not results:
        return []
    best = max(results.values())
    return sorted(position for position, result in results.items() if result == best)


def _matchup_from_types(hand_types: Dict[int, str]) -> str:
    return " vs ".join(sorted(hand_types.values()))


def _compute_contributions(
    standard_rewards: Dict[int, float],
    reward_inferred_winners: List[int],
    pot_size: float,
) -> Dict[int, float]:
    if not reward_inferred_winners:
        raise ValueError("Cannot compute contributions without at least one winner")

    share = pot_size / len(reward_inferred_winners)
    contributions: Dict[int, float] = {}
    for position, reward in standard_rewards.items():
        standard_payout = share if position in reward_inferred_winners else 0.0
        contributions[position] = standard_payout - reward
    return contributions


def _recompute_rewards(
    contributions: Dict[int, float],
    winners: List[int],
    pot_size: float,
) -> Dict[int, float | int]:
    share = pot_size / len(winners) if winners else 0.0
    rewards: Dict[int, float | int] = {}
    for position, contribution in contributions.items():
        payout = share if position in winners else 0.0
        rewards[position] = clean_number(payout - contribution)
    return rewards


def reevaluate_hand(
    records: List[Dict[str, Any]],
    standard_evaluator: StandardHandEvaluator,
    infinite_evaluator: InfiniteHandEvaluator,
    *,
    heads_up_only: bool,
) -> HandAuditReport:
    hand_id = int(records[0]["hand_id"])
    players, board, final_pot = reconstruct_hand(records)
    report = HandAuditReport(hand_id=hand_id, board=board)

    if heads_up_only and len(players) != 2:
        report.skip_reason = "not_heads_up"
        return report

    live_players = {position: player for position, player in players.items() if not player.folded}
    if len(live_players) < 2 or len(board) < 5:
        report.skip_reason = "non_showdown"
        return report

    if any(any(card == -1 for card in player.hole_cards) for player in live_players.values()):
        report.showdown = True
        report.skip_reason = "missing_hole_cards"
        return report

    report.showdown = True

    standard_results = {
        position: standard_evaluator.evaluate([idx_to_card(card) for card in player.hole_cards + board])
        for position, player in live_players.items()
    }
    infinite_results = {
        position: infinite_evaluator.evaluate([idx_to_card(card) for card in player.hole_cards + board])
        for position, player in live_players.items()
    }

    report.standard_winners = _winners_from_results(standard_results)
    report.infinite_winners = _winners_from_results(infinite_results)

    standard_rewards = {
        position: float(player.standard_reward if player.standard_reward is not None else 0.0)
        for position, player in live_players.items()
    }
    report.standard_rewards = {
        position: clean_number(reward) for position, reward in standard_rewards.items()
    }

    max_reward = max(standard_rewards.values())
    report.reward_inferred_standard_winners = sorted(
        position for position, reward in standard_rewards.items() if reward == max_reward
    )
    report.standard_winner_mismatch = set(report.standard_winners) != set(report.reward_inferred_standard_winners)

    report.standard_hand_types = {
        position: result.rank.name for position, result in standard_results.items()
    }
    report.infinite_hand_types = {
        position: result.rank.name for position, result in infinite_results.items()
    }
    report.standard_descriptions = {
        position: result.description for position, result in standard_results.items()
    }
    report.infinite_descriptions = {
        position: result.description for position, result in infinite_results.items()
    }
    report.best_hand_changed = {
        position: report.standard_hand_types[position] != report.infinite_hand_types[position]
        for position in live_players
    }
    report.standard_matchup = _matchup_from_types(report.standard_hand_types)
    report.infinite_matchup = _matchup_from_types(report.infinite_hand_types)
    report.winner_flipped = set(report.standard_winners) != set(report.infinite_winners)

    contributions = _compute_contributions(
        standard_rewards,
        report.reward_inferred_standard_winners,
        final_pot,
    )
    report.infinite_rewards = _recompute_rewards(
        contributions,
        report.infinite_winners,
        final_pot,
    )
    report.showdown_reevaluated = True
    return report


def augment_record(record: Dict[str, Any], report: HandAuditReport) -> Dict[str, Any]:
    augmented = dict(record)
    source_variant_tag = record.get("variant_tag", "unknown")
    actor_position = int(record["state_vector"]["position"])

    augmented["source_variant_tag"] = source_variant_tag
    augmented["variant_tag"] = f"{source_variant_tag}_{STANDARD_AUGMENTED_SUFFIX}"
    augmented["standard_reward"] = record.get("reward")
    augmented["infinite_reward"] = (
        report.infinite_rewards.get(actor_position, record.get("reward"))
        if report.showdown_reevaluated
        else record.get("reward")
    )
    augmented["showdown_reevaluated"] = report.showdown_reevaluated
    augmented["winner_flipped"] = report.winner_flipped if report.showdown_reevaluated else False
    augmented["standard_winners"] = report.standard_winners if report.showdown_reevaluated else []
    augmented["infinite_winners"] = report.infinite_winners if report.showdown_reevaluated else []
    augmented["standard_best_hand_type"] = (
        report.standard_hand_types.get(actor_position) if report.showdown_reevaluated else None
    )
    augmented["infinite_best_hand_type"] = (
        report.infinite_hand_types.get(actor_position) if report.showdown_reevaluated else None
    )
    augmented["standard_best_hand_description"] = (
        report.standard_descriptions.get(actor_position) if report.showdown_reevaluated else None
    )
    augmented["infinite_best_hand_description"] = (
        report.infinite_descriptions.get(actor_position) if report.showdown_reevaluated else None
    )
    augmented["best_hand_changed"] = (
        report.best_hand_changed.get(actor_position, False) if report.showdown_reevaluated else False
    )
    return augmented


def build_summary(
    reports: List[HandAuditReport],
    original_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    n_total = len(reports)
    showdown_hands = [report for report in reports if report.showdown]
    reevaluated = [report for report in reports if report.showdown_reevaluated]
    flipped = [report for report in reevaluated if report.winner_flipped]

    hand_type_transitions = Counter()
    changed_transitions = Counter()
    winner_flip_matchups_standard = Counter()
    winner_flip_matchups_infinite = Counter()
    skipped_reasons = Counter(report.skip_reason for report in reports if report.skip_reason)

    for report in reevaluated:
        for position, standard_type in report.standard_hand_types.items():
            infinite_type = report.infinite_hand_types[position]
            transition = f"{standard_type}->{infinite_type}"
            hand_type_transitions[transition] += 1
            if standard_type != infinite_type:
                changed_transitions[transition] += 1

        if report.winner_flipped:
            winner_flip_matchups_standard[report.standard_matchup] += 1
            winner_flip_matchups_infinite[report.infinite_matchup] += 1

    variant_counts = Counter(record.get("variant_tag", "unknown") for record in original_records)

    impossible_flip_matchups = [
        matchup
        for matchup in winner_flip_matchups_standard
        if not set(matchup.split(" vs ")).issubset(CHANGING_HAND_TYPES)
    ]

    return {
        "n_total_hands": n_total,
        "n_showdown_hands": len(showdown_hands),
        "n_reevaluated_showdowns": len(reevaluated),
        "n_flipped_hands": len(flipped),
        "flip_rate_over_reevaluated_showdowns": (
            len(flipped) / len(reevaluated) if reevaluated else 0.0
        ),
        "flip_rate_over_all_hands": len(flipped) / n_total if n_total else 0.0,
        "skip_reasons": dict(skipped_reasons),
        "standard_winner_mismatches": sum(1 for report in reevaluated if report.standard_winner_mismatch),
        "source_variant_counts": dict(variant_counts),
        "hand_type_transitions": dict(hand_type_transitions.most_common()),
        "changed_hand_type_transitions": dict(changed_transitions.most_common()),
        "winner_flip_matchups_standard": dict(winner_flip_matchups_standard.most_common()),
        "winner_flip_matchups_infinite": dict(winner_flip_matchups_infinite.most_common()),
        "impossible_flip_matchups": impossible_flip_matchups,
    }


def write_outputs(
    output_prefix: Path,
    reports: List[HandAuditReport],
    summary: Dict[str, Any],
    augmented_records: Optional[List[Dict[str, Any]]],
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    per_hand_path = output_prefix.with_name(output_prefix.name + ".per_hand.jsonl")
    with per_hand_path.open("w", encoding="utf-8") as handle:
        for report in reports:
            handle.write(json.dumps(asdict(report)) + "\n")

    summary_path = output_prefix.with_name(output_prefix.name + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if augmented_records is not None:
        augmented_path = output_prefix.with_name(output_prefix.name + ".augmented.json")
        augmented_path.write_text(json.dumps(augmented_records, indent=2), encoding="utf-8")
        print(f"Augmented dataset: {augmented_path}")

    print(f"Per-hand report:   {per_hand_path}")
    print(f"Summary:           {summary_path}")


def scan_dataset(
    input_path: Path,
    output_prefix: Path,
    *,
    write_augmented: bool,
    heads_up_only: bool,
    max_hands: Optional[int],
) -> None:
    records = load_records(input_path)
    hand_order, by_hand = group_records_by_hand(records, max_hands=max_hands)

    standard_evaluator = StandardHandEvaluator()
    infinite_evaluator = InfiniteHandEvaluator()
    reports_by_hand: Dict[int, HandAuditReport] = {}
    reports: List[HandAuditReport] = []
    selected_hand_ids = set(hand_order)

    for hand_id in hand_order:
        report = reevaluate_hand(
            by_hand[hand_id],
            standard_evaluator,
            infinite_evaluator,
            heads_up_only=heads_up_only,
        )
        reports_by_hand[hand_id] = report
        reports.append(report)

    augmented_records: Optional[List[Dict[str, Any]]] = None
    if write_augmented:
        augmented_records = [
            augment_record(record, reports_by_hand[int(record["hand_id"])])
            for record in records
            if int(record["hand_id"]) in selected_hand_ids
        ]

    selected_records = [record for record in records if int(record["hand_id"]) in selected_hand_ids]
    summary = build_summary(reports, selected_records)
    write_outputs(output_prefix, reports, summary, augmented_records)

    print(f"Total hands:                {summary['n_total_hands']}")
    print(f"Showdown hands:             {summary['n_showdown_hands']}")
    print(f"Reevaluated showdowns:      {summary['n_reevaluated_showdowns']}")
    print(f"Winner flipped:             {summary['n_flipped_hands']}")
    if summary["n_reevaluated_showdowns"]:
        print(
            f"Flip rate over showdowns:   "
            f"{summary['flip_rate_over_reevaluated_showdowns'] * 100:.2f}%"
        )
    if summary["standard_winner_mismatches"]:
        print(f"Standard winner mismatches: {summary['standard_winner_mismatches']}")
    if summary["impossible_flip_matchups"]:
        print("Warning: found flip matchups outside Straight/Flush/Full House cluster")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input standard-rules trajectory JSON")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reevaluation_report"),
        help="Output file prefix (default: reevaluation_report)",
    )
    parser.add_argument(
        "--write-augmented",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the augmented per-decision dataset (default: true)",
    )
    parser.add_argument(
        "--heads-up-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip hands that are not exactly two-player (default: true)",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=None,
        help="Limit processing to the first N hands in file order",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    scan_dataset(
        arguments.input,
        arguments.out,
        write_augmented=arguments.write_augmented,
        heads_up_only=arguments.heads_up_only,
        max_hands=arguments.max_hands,
    )
