#!/usr/bin/env python3
"""
Mix two hand-level JSON poker datasets into one shuffled output file.

The mixer keeps all rows for a given `hand_id` together and preserves the
within-hand order by sorting each hand on `step_in_hand`. It then selects a
user-specified total number of hands from the two input datasets, attempting to
balance the final output to roughly 50/50 by row count while respecting the
hand grouping constraint.

Example:
    python3 mix_datasets.py \
        data/infinite_holdem_training_data.json \
        data/irc_holdem_199806.json \
        --total-hands 1000 \
        --output data/mixed_dataset.json \
        --seed 7 \
        --indent 2
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Row = Dict[str, object]
Hand = List[Row]


def load_rows(path: Path) -> List[Row]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON array")
    return data


def validate_rows(rows: Sequence[Row], path: Path) -> None:
    if not rows:
        raise ValueError(f"{path} is empty")

    required_keys = {"hand_id", "step_in_hand"}
    first_missing = required_keys - set(rows[0].keys())
    if first_missing:
        missing = ", ".join(sorted(first_missing))
        raise ValueError(f"{path} is missing required keys: {missing}")

    for index, row in enumerate(rows):
        missing = required_keys - set(row.keys())
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{path} row {index} is missing required keys: {missing_text}")


def schema_signature(rows: Sequence[Row]) -> Tuple[str, ...]:
    return tuple(sorted(rows[0].keys()))


def group_rows_by_hand(rows: Sequence[Row]) -> List[Hand]:
    grouped: Dict[object, List[Row]] = {}
    hand_order: List[object] = []

    for row in rows:
        hand_id = row["hand_id"]
        if hand_id not in grouped:
            grouped[hand_id] = []
            hand_order.append(hand_id)
        grouped[hand_id].append(dict(row))

    hands: List[Hand] = []
    for hand_id in hand_order:
        hand_rows = grouped[hand_id]
        hand_rows.sort(key=lambda row: int(row["step_in_hand"]))
        hands.append(hand_rows)
    return hands


def ensure_disjoint_hand_ids(hands_a: Sequence[Hand], hands_b: Sequence[Hand]) -> None:
    ids_a = {hand[0]["hand_id"] for hand in hands_a}
    ids_b = {hand[0]["hand_id"] for hand in hands_b}
    overlap = ids_a & ids_b
    if overlap:
        sample = ", ".join(str(value) for value in list(sorted(overlap))[:5])
        raise ValueError(
            "Input datasets share hand_id values, which would break grouping in the mixed output. "
            f"Example overlaps: {sample}"
        )


def choose_hand_source(
    *,
    next_a_size: Optional[int],
    next_b_size: Optional[int],
    rows_from_a: int,
    rows_from_b: int,
    hands_from_a: int,
    hands_from_b: int,
    rng: random.Random,
) -> str:
    if next_a_size is None:
        return "b"
    if next_b_size is None:
        return "a"

    diff_if_a = abs((rows_from_a + next_a_size) - rows_from_b)
    diff_if_b = abs(rows_from_a - (rows_from_b + next_b_size))

    if diff_if_a < diff_if_b:
        return "a"
    if diff_if_b < diff_if_a:
        return "b"

    if hands_from_a < hands_from_b:
        return "a"
    if hands_from_b < hands_from_a:
        return "b"

    return rng.choice(["a", "b"])


def select_hands(
    hands_a: Sequence[Hand],
    hands_b: Sequence[Hand],
    *,
    total_hands: int,
    rng: random.Random,
) -> Tuple[List[Hand], Dict[str, int]]:
    shuffled_a = list(hands_a)
    shuffled_b = list(hands_b)
    rng.shuffle(shuffled_a)
    rng.shuffle(shuffled_b)

    selected: List[Hand] = []
    hands_from_a = 0
    hands_from_b = 0
    rows_from_a = 0
    rows_from_b = 0
    index_a = 0
    index_b = 0

    while len(selected) < total_hands:
        next_a_size = len(shuffled_a[index_a]) if index_a < len(shuffled_a) else None
        next_b_size = len(shuffled_b[index_b]) if index_b < len(shuffled_b) else None

        source = choose_hand_source(
            next_a_size=next_a_size,
            next_b_size=next_b_size,
            rows_from_a=rows_from_a,
            rows_from_b=rows_from_b,
            hands_from_a=hands_from_a,
            hands_from_b=hands_from_b,
            rng=rng,
        )

        if source == "a":
            hand = shuffled_a[index_a]
            index_a += 1
            hands_from_a += 1
            rows_from_a += len(hand)
        else:
            hand = shuffled_b[index_b]
            index_b += 1
            hands_from_b += 1
            rows_from_b += len(hand)

        selected.append(hand)

    rng.shuffle(selected)

    stats = {
        "hands_from_a": hands_from_a,
        "hands_from_b": hands_from_b,
        "rows_from_a": rows_from_a,
        "rows_from_b": rows_from_b,
    }
    return selected, stats


def flatten_hands(hands: Iterable[Hand]) -> List[Row]:
    flat_rows: List[Row] = []
    for hand in hands:
        flat_rows.extend(hand)
    return flat_rows


def summarize_variants(rows: Sequence[Row]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[str(row.get("variant_tag", "unknown"))] += 1
    return dict(counter)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mix two JSON poker datasets while preserving hand grouping."
    )
    parser.add_argument("dataset_a", help="First input JSON dataset path.")
    parser.add_argument("dataset_b", help="Second input JSON dataset path.")
    parser.add_argument(
        "--total-hands",
        type=int,
        required=True,
        help="Total number of hands to include in the mixed output.",
    )
    parser.add_argument(
        "--output",
        default="data/mixed_dataset.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for hand selection and shuffling.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Pretty-print indentation. Omit for compact JSON.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.total_hands <= 0:
        raise ValueError("--total-hands must be positive")

    path_a = Path(args.dataset_a)
    path_b = Path(args.dataset_b)
    output_path = Path(args.output)

    rows_a = load_rows(path_a)
    rows_b = load_rows(path_b)
    validate_rows(rows_a, path_a)
    validate_rows(rows_b, path_b)

    if schema_signature(rows_a) != schema_signature(rows_b):
        raise ValueError(
            "Input datasets do not share the same top-level row schema. "
            "Regenerate them into the same format before mixing."
        )

    hands_a = group_rows_by_hand(rows_a)
    hands_b = group_rows_by_hand(rows_b)
    ensure_disjoint_hand_ids(hands_a, hands_b)

    total_available_hands = len(hands_a) + len(hands_b)
    if args.total_hands > total_available_hands:
        raise ValueError(
            f"Requested {args.total_hands} hands, but only {total_available_hands} are available "
            f"({len(hands_a)} in dataset A, {len(hands_b)} in dataset B)."
        )

    rng = random.Random(args.seed)
    selected_hands, stats = select_hands(
        hands_a,
        hands_b,
        total_hands=args.total_hands,
        rng=rng,
    )
    mixed_rows = flatten_hands(selected_hands)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(mixed_rows, handle, indent=args.indent)
        if args.indent is not None:
            handle.write("\n")

    total_rows = len(mixed_rows)
    rows_from_a = stats["rows_from_a"]
    rows_from_b = stats["rows_from_b"]
    pct_a = (rows_from_a / total_rows * 100.0) if total_rows else 0.0
    pct_b = (rows_from_b / total_rows * 100.0) if total_rows else 0.0

    print("Mixed dataset written.")
    print(f"Output path: {output_path}")
    print(f"Total hands: {len(selected_hands)}")
    print(f"Total rows: {total_rows}")
    print(
        f"Dataset A ({path_a.name}): hands={stats['hands_from_a']}, rows={rows_from_a}, "
        f"share={pct_a:.1f}%"
    )
    print(
        f"Dataset B ({path_b.name}): hands={stats['hands_from_b']}, rows={rows_from_b}, "
        f"share={pct_b:.1f}%"
    )
    print(f"Variant breakdown: {summarize_variants(mixed_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
