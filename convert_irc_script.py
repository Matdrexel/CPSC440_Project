#!/usr/bin/env python3
"""
Convert fixed-limit IRC hold'em archives into the same per-decision JSON shape
used by `pokerkit_examples/simulate_games.py`.

The converter is intentionally conservative:
- it only processes heads-up hands by default, because the common schema is
  heads-up (`stack`, `opponent_stack`, two-entry `current_bets`, etc.),
- it only targets fixed-limit hold'em style archives where street bet sizing is
  recoverable from the action strings,
- it only emits rows for players whose private cards are known in the IRC logs.

Example:
    python3 convert_irc_script.py \
        --input CPSC440_Project/IRCdata \
        --output CPSC440_Project/data/irc_standard_holdem_training_data.json \
        --max-archives 1 \
        --max-hands 5000 \
        --indent 2
"""

from __future__ import annotations

import argparse
import collections
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, TextIO, Tuple


RANKS = "23456789TJQKA"
SUITS = "cdhs"
CARD_TO_INDEX = {f"{rank}{suit}": rank_index * 4 + suit_index for rank_index, rank in enumerate(RANKS) for suit_index, suit in enumerate(SUITS)}

# These archives are fixed-limit Texas hold'em style datasets whose action
# strings can be replayed into the heads-up schema used by `simulate_games.py`.
SUPPORTED_ARCHIVE_PREFIXES = (
    "holdem.",
    "holdem1.",
    "holdem2.",
    "holdem3.",
    "h1-nobots.",
    "botsonly.",
)

MAX_LIMIT_BETS_PER_ROUND = 4
HAND_ID_ARCHIVE_MULTIPLIER = 10**12
STANDARD_VARIANT_TAG = "standard_holdem_irc"


@dataclass(frozen=True)
class HdbRecord:
    timestamp: int
    hand_number: int
    num_players: int
    board_cards: List[str]
    preflop_summary: Tuple[int, int]
    flop_summary: Tuple[int, int]
    turn_summary: Tuple[int, int]
    final_summary: Tuple[int, int]


@dataclass(frozen=True)
class PlayerRecord:
    name: str
    timestamp: int
    num_players: int
    position: int
    actions: List[str]
    bankroll: int
    paid: int
    won: int
    hole_cards: List[str]

    @property
    def reward(self) -> int:
        return self.won - self.paid


def cards_to_count_encoding(cards: Sequence[str]) -> List[int]:
    counts = [0] * 52
    for card in cards:
        if card in CARD_TO_INDEX:
            counts[CARD_TO_INDEX[card]] += 1
    return counts


def parse_summary(value: str) -> Tuple[int, int]:
    players_left, pot_size = value.split("/", 1)
    return int(players_left), int(pot_size)


def parse_hdb_line(line: str) -> Optional[HdbRecord]:
    parts = line.split()
    if len(parts) < 8:
        return None

    board_cards = parts[8:]
    return HdbRecord(
        timestamp=int(parts[0]),
        hand_number=int(parts[2]),
        num_players=int(parts[3]),
        board_cards=board_cards,
        preflop_summary=parse_summary(parts[4]),
        flop_summary=parse_summary(parts[5]),
        turn_summary=parse_summary(parts[6]),
        final_summary=parse_summary(parts[7]),
    )


def parse_pdb_line(line: str) -> Optional[PlayerRecord]:
    parts = line.split()
    if len(parts) < 11:
        return None

    hole_cards: List[str] = []
    if len(parts) >= 13 and parts[-2] in CARD_TO_INDEX and parts[-1] in CARD_TO_INDEX:
        hole_cards = [parts[-2], parts[-1]]
    elif len(parts) >= 12 and len(parts[11]) == 4 and "?" not in parts[11]:
        hole_cards = [parts[11][:2], parts[11][2:4]]

    return PlayerRecord(
        name=parts[0],
        timestamp=int(parts[1]),
        num_players=int(parts[2]),
        position=int(parts[3]),
        actions=[parts[4], parts[5], parts[6], parts[7]],
        bankroll=int(parts[8]),
        paid=int(parts[9]),
        won=int(parts[10]),
        hole_cards=hole_cards,
    )


def cleaned_actions(action_string: str, *, strip_blind: bool = False) -> str:
    cleaned = action_string.replace("-", "")
    if strip_blind and cleaned.startswith("B"):
        cleaned = cleaned[1:]
    return cleaned


def is_supported_archive(archive_path: Path) -> bool:
    return archive_path.name.startswith(SUPPORTED_ARCHIVE_PREFIXES)


def iter_archives(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    return sorted(path for path in input_path.glob("*.tgz") if is_supported_archive(path))


def infer_blinds(pdb_grouped: Dict[int, List[PlayerRecord]]) -> Tuple[int, int]:
    blind_pairs: collections.Counter[Tuple[int, int]] = collections.Counter()

    for records in pdb_grouped.values():
        if len(records) != 2:
            continue

        by_pos = {record.position: record for record in records}
        if set(by_pos) != {1, 2}:
            continue

        pre1 = by_pos[1].actions[0]
        pre2 = by_pos[2].actions[0]
        if not (pre1.startswith("B") and pre2.startswith("B")):
            continue

        rest1 = cleaned_actions(pre1, strip_blind=True)
        rest2 = cleaned_actions(pre2, strip_blind=True)
        later_streets_empty = all(street == "-" for record in records for street in record.actions[1:])

        if later_streets_empty and rest1 in {"", "f"} and rest2 in {"", "f"}:
            positive_paid = sorted(record.paid for record in records if record.paid > 0)
            if len(positive_paid) == 2 and positive_paid[0] < positive_paid[1]:
                blind_pairs[(positive_paid[0], positive_paid[1])] += 1

    if blind_pairs:
        return blind_pairs.most_common(1)[0][0]

    positive_paid = sorted({record.paid for records in pdb_grouped.values() for record in records if record.paid > 0})
    if len(positive_paid) >= 2:
        return positive_paid[0], positive_paid[1]

    raise ValueError("Unable to infer small blind / big blind from archive")


def street_order(street: str) -> List[int]:
    # In the heads-up IRC logs, the action strings are ordered by IRC position
    # (`1` then `2`) on every street. Using the standard hold'em postflop order
    # here mis-replays strings like `kc` into impossible states.
    _ = street
    return [1, 2]


def limit_bet_size(big_blind: int, street: str) -> int:
    return big_blind if street in {"preflop", "flop"} else big_blind * 2


def expected_final_pot(hdb_record: HdbRecord) -> int:
    return hdb_record.final_summary[1]


def maybe_validate_pot(actual_pot: int, expected_pot: int) -> bool:
    return expected_pot == 0 or actual_pot == expected_pot


def legal_actions_for_state(
    *,
    pot_size: int,
    current_bets: Dict[int, int],
    actor_pos: int,
    total_contributed: Dict[int, int],
    bankrolls: Dict[int, int],
    bet_level: int,
    bet_size: int,
) -> Tuple[List[str], Optional[int], Optional[int]]:
    current_high_bet = max(current_bets.values())
    to_call = current_high_bet - current_bets[actor_pos]
    stack_remaining = bankrolls[actor_pos] - total_contributed[actor_pos]
    can_raise = bet_level < MAX_LIMIT_BETS_PER_ROUND and stack_remaining > to_call

    if to_call == 0:
        legal = ["check"]
    else:
        legal = ["fold", "call"]

    min_raise_to: Optional[int] = None
    max_raise_to: Optional[int] = None
    if can_raise:
        legal.append("raise")
        min_raise_to = current_high_bet + bet_size if current_high_bet > 0 else bet_size
        max_raise_to = min_raise_to

    return legal, min_raise_to, max_raise_to


def make_example(
    *,
    hand_id: int,
    step_in_hand: int,
    street: str,
    actor_pos: int,
    player_records: Dict[int, PlayerRecord],
    bankrolls: Dict[int, int],
    total_contributed: Dict[int, int],
    current_bets: Dict[int, int],
    pot_size: int,
    board_cards: List[str],
    legal_actions: List[str],
    min_raise_to: Optional[int],
    max_raise_to: Optional[int],
    action: str,
    raise_amount: Optional[int],
) -> Optional[Dict]:
    actor_record = player_records[actor_pos]
    if len(actor_record.hole_cards) != 2:
        return None

    opponent_pos = 2 if actor_pos == 1 else 1
    position_to_index = {1: 0, 2: 1}

    stack = bankrolls[actor_pos] - total_contributed[actor_pos]
    opponent_stack = bankrolls[opponent_pos] - total_contributed[opponent_pos]
    current_high_bet = max(current_bets.values())
    to_call = current_high_bet - current_bets[actor_pos]

    return {
        "hole_cards": list(actor_record.hole_cards),
        "board_cards": list(board_cards),
        "hole_card_counts": cards_to_count_encoding(actor_record.hole_cards),
        "board_card_counts": cards_to_count_encoding(board_cards),
        "pot_size": pot_size,
        "stack": stack,
        "opponent_stack": opponent_stack,
        "position": position_to_index[actor_pos],
        "street": street,
        "to_call": to_call,
        "current_bets": [current_bets[1], current_bets[2]],
        "legal_actions": list(legal_actions),
        "min_raise_to": min_raise_to,
        "max_raise_to": max_raise_to,
        "action": action,
        "raise_amount": raise_amount,
        "reward": actor_record.reward,
        "hand_id": hand_id,
        "step_in_hand": step_in_hand,
        "has_duplicates": False,
        "variant_tag": STANDARD_VARIANT_TAG,
    }


def convert_heads_up_hand(
    *,
    hand_id: int,
    hdb_record: HdbRecord,
    player_records: List[PlayerRecord],
    small_blind: int,
    big_blind: int,
) -> Optional[List[Dict]]:
    if len(player_records) != 2:
        return None

    records_by_pos = {record.position: record for record in player_records}
    if set(records_by_pos) != {1, 2}:
        return None

    bankrolls = {1: records_by_pos[1].bankroll, 2: records_by_pos[2].bankroll}
    total_contributed = {1: small_blind, 2: big_blind}
    pot_size = small_blind + big_blind
    board_cards: List[str] = []
    examples: List[Dict] = []
    step_in_hand = 0
    hand_over = False

    street_names = ["preflop", "flop", "turn", "river"]
    street_validations = {
        "preflop": hdb_record.preflop_summary[1],
        "flop": hdb_record.flop_summary[1],
        "turn": hdb_record.turn_summary[1],
    }

    for street_index, street in enumerate(street_names):
        if hand_over:
            break

        if street == "flop":
            if len(hdb_record.board_cards) < 3:
                break
            board_cards.extend(hdb_record.board_cards[:3])
        elif street == "turn":
            if len(hdb_record.board_cards) < 4:
                break
            board_cards.append(hdb_record.board_cards[3])
        elif street == "river":
            if len(hdb_record.board_cards) < 5:
                break
            board_cards.append(hdb_record.board_cards[4])

        if street == "preflop":
            current_bets = {1: small_blind, 2: big_blind}
            bet_level = 1
        else:
            current_bets = {1: 0, 2: 0}
            bet_level = 0

        bet_size = limit_bet_size(big_blind, street)
        pending = street_order(street)[:]
        actions_by_pos = {
            1: cleaned_actions(records_by_pos[1].actions[street_index], strip_blind=(street == "preflop")),
            2: cleaned_actions(records_by_pos[2].actions[street_index], strip_blind=(street == "preflop")),
        }
        action_index = {1: 0, 2: 0}

        while pending:
            actor_pos = pending[0]
            actor_actions = actions_by_pos[actor_pos]
            if action_index[actor_pos] >= len(actor_actions):
                break

            action_char = actor_actions[action_index[actor_pos]]
            action_index[actor_pos] += 1
            current_high_bet = max(current_bets.values())
            to_call = current_high_bet - current_bets[actor_pos]
            legal_actions, min_raise_to, max_raise_to = legal_actions_for_state(
                pot_size=pot_size,
                current_bets=current_bets,
                actor_pos=actor_pos,
                total_contributed=total_contributed,
                bankrolls=bankrolls,
                bet_level=bet_level,
                bet_size=bet_size,
            )

            if action_char == "f":
                action = "fold"
                raise_amount = None
            elif action_char in {"k", "c"}:
                action = "call" if to_call > 0 else "check"
                raise_amount = None
            elif action_char in {"b", "r", "A"}:
                action = "raise"
                raise_amount = min_raise_to
            else:
                return None

            if action not in legal_actions:
                return None

            example = make_example(
                hand_id=hand_id,
                step_in_hand=step_in_hand,
                street=street,
                actor_pos=actor_pos,
                player_records=records_by_pos,
                bankrolls=bankrolls,
                total_contributed=total_contributed,
                current_bets=current_bets,
                pot_size=pot_size,
                board_cards=board_cards,
                legal_actions=legal_actions,
                min_raise_to=min_raise_to,
                max_raise_to=max_raise_to,
                action=action,
                raise_amount=raise_amount,
            )
            if example is not None:
                examples.append(example)

            if action == "fold":
                hand_over = True
                break

            if action in {"check", "call"}:
                additional = min(to_call, bankrolls[actor_pos] - total_contributed[actor_pos])
                pot_size += additional
                current_bets[actor_pos] += additional
                total_contributed[actor_pos] += additional
                pending.pop(0)
            else:
                total_to = min_raise_to if min_raise_to is not None else current_high_bet + bet_size
                if total_to is None:
                    return None
                additional = min(total_to - current_bets[actor_pos], bankrolls[actor_pos] - total_contributed[actor_pos])
                pot_size += additional
                current_bets[actor_pos] += additional
                total_contributed[actor_pos] += additional
                bet_level += 1
                pending = [2 if actor_pos == 1 else 1]

            step_in_hand += 1

        if hand_over:
            break

        if street in street_validations and not maybe_validate_pot(pot_size, street_validations[street]):
            return None

    if not maybe_validate_pot(pot_size, expected_final_pot(hdb_record)):
        return None

    paid_total = sum(record.paid for record in records_by_pos.values())
    won_total = sum(record.won for record in records_by_pos.values())
    if paid_total != won_total:
        return None

    if sum(record.reward for record in records_by_pos.values()) != 0:
        return None

    return examples


def parse_archive(
    archive_path: Path,
    *,
    archive_index: int,
    max_hands: Optional[int],
) -> Tuple[List[Dict], Dict[str, int]]:
    stats = {
        "hands_seen": 0,
        "hands_converted": 0,
        "rows_emitted": 0,
        "hands_skipped": 0,
    }

    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        hdb_member = next((member for member in members if member.name.endswith("/hdb")), None)
        if hdb_member is None:
            return [], stats

        hdb_records: Dict[int, HdbRecord] = {}
        hdb_file = archive.extractfile(hdb_member)
        if hdb_file is None:
            return [], stats

        for raw_line in hdb_file:
            record = parse_hdb_line(raw_line.decode("utf-8", "ignore").strip())
            if record is None or record.num_players != 2:
                continue
            hdb_records[record.timestamp] = record
            if max_hands is not None and len(hdb_records) >= max_hands:
                break

        if not hdb_records:
            return [], stats

        target_timestamps = set(hdb_records)
        pdb_grouped: DefaultDict[int, List[PlayerRecord]] = collections.defaultdict(list)
        pdb_members = [member for member in members if "/pdb/pdb." in member.name]

        for member in pdb_members:
            pdb_file = archive.extractfile(member)
            if pdb_file is None:
                continue
            for raw_line in pdb_file:
                record = parse_pdb_line(raw_line.decode("utf-8", "ignore").strip())
                if record is None or record.timestamp not in target_timestamps:
                    continue
                pdb_grouped[record.timestamp].append(record)

        small_blind, big_blind = infer_blinds(pdb_grouped)
        examples: List[Dict] = []

        for timestamp in sorted(target_timestamps):
            stats["hands_seen"] += 1
            player_records = pdb_grouped.get(timestamp, [])
            if len(player_records) != 2:
                stats["hands_skipped"] += 1
                continue

            unique_hand_id = archive_index * HAND_ID_ARCHIVE_MULTIPLIER + timestamp
            hand_examples = convert_heads_up_hand(
                hand_id=unique_hand_id,
                hdb_record=hdb_records[timestamp],
                player_records=player_records,
                small_blind=small_blind,
                big_blind=big_blind,
            )
            if not hand_examples:
                stats["hands_skipped"] += 1
                continue

            examples.extend(hand_examples)
            stats["hands_converted"] += 1
            stats["rows_emitted"] += len(hand_examples)

    return examples, stats


def append_json_examples(
    handle: TextIO,
    examples: List[Dict],
    *,
    indent: Optional[int],
    first: bool,
) -> Tuple[bool, int]:
    count = 0
    for example in examples:
        if first:
            first = False
            if indent is not None:
                handle.write("\n")
        else:
            handle.write(",\n" if indent is not None else ",")

        json.dump(example, handle, indent=indent)
        count += 1

    return first, count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert fixed-limit IRC hold'em archives to the common JSON training schema.")
    parser.add_argument(
        "--input",
        default="CPSC440_Project/IRCdata",
        help="IRC archive directory or a single .tgz archive path.",
    )
    parser.add_argument(
        "--output",
        default="CPSC440_Project/data/irc_standard_holdem_training_data.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--max-archives",
        type=int,
        default=None,
        help="Optional cap on how many archives to process.",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=None,
        help="Optional cap on how many heads-up hands to read from each archive.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Pretty-print indentation. Omit for compact JSON.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    archives = iter_archives(input_path)
    if args.max_archives is not None:
        archives = archives[: args.max_archives]

    if not archives:
        raise FileNotFoundError(f"No supported IRC archives found under {input_path}")

    aggregate_stats = collections.Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("[")
        first = True

        for archive_index, archive_path in enumerate(archives, start=1):
            print(f"Processing {archive_path.name}...")
            examples, stats = parse_archive(
                archive_path,
                archive_index=archive_index,
                max_hands=args.max_hands,
            )
            first, archive_rows = append_json_examples(
                handle,
                examples,
                indent=args.indent,
                first=first,
            )
            row_count += archive_rows
            aggregate_stats.update(stats)
            print(
                f"  heads-up hands seen={stats['hands_seen']}, converted={stats['hands_converted']}, "
                f"skipped={stats['hands_skipped']}, rows={stats['rows_emitted']}"
            )

        if args.indent is not None and not first:
            handle.write("\n")
        handle.write("]")
        if args.indent is not None:
            handle.write("\n")

    print("\nConversion complete.")
    print(f"Archives processed: {len(archives)}")
    print(f"Hands seen: {aggregate_stats['hands_seen']}")
    print(f"Hands converted: {aggregate_stats['hands_converted']}")
    print(f"Hands skipped: {aggregate_stats['hands_skipped']}")
    print(f"Rows written: {row_count}")
    print(f"Saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
