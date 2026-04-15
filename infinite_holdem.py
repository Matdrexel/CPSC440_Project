"""
infinite_holdem.py - Infinite Texas Hold'em (cards dealt with replacement).

Phase 1 focuses on a heads-up research engine with:
- replacement dealing,
- duplicate-aware hand evaluation,
- straight > flush ranking,
- real heads-up action flow with reopened action after raises,
- zero-sum pot settlement.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from typing import List, Optional, Tuple


RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS = ["c", "d", "h", "s"]
RANK_VALUES = {rank: index for index, rank in enumerate(RANKS)}
RANK_NAMES = {
    "2": "Twos",
    "3": "Threes",
    "4": "Fours",
    "5": "Fives",
    "6": "Sixes",
    "7": "Sevens",
    "8": "Eights",
    "9": "Nines",
    "T": "Tens",
    "J": "Jacks",
    "Q": "Queens",
    "K": "Kings",
    "A": "Aces",
}
STREETS = ("preflop", "flop", "turn", "river", "showdown")


@dataclass(frozen=True)
class Card:
    """A playing card with rank and suit."""

    rank: str
    suit: str

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        return str(self)

    @property
    def rank_value(self) -> int:
        return RANK_VALUES[self.rank]

    @classmethod
    def from_string(cls, value: str) -> "Card":
        return cls(rank=value[0], suit=value[1])


def card_to_index(card: Card) -> int:
    """Convert a card to a stable 0-51 index."""

    return RANK_VALUES[card.rank] * 4 + SUITS.index(card.suit)


def index_to_card(index: int) -> Card:
    """Convert a 0-51 index back to a card."""

    rank_index = index // 4
    suit_index = index % 4
    return Card(rank=RANKS[rank_index], suit=SUITS[suit_index])


class InfiniteDeck:
    """A deck that samples from the full 52-card space with replacement."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def deal(self, n: int = 1) -> List[Card]:
        return [index_to_card(self.rng.randint(0, 51)) for _ in range(n)]

    def deal_specific(self, card_strings: List[str]) -> List[Card]:
        return [Card.from_string(value) for value in card_strings]


class HandRank(IntEnum):
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    FULL_HOUSE = 5
    FLUSH = 6
    STRAIGHT = 7
    FOUR_OF_A_KIND = 8
    FIVE_OF_A_KIND = 9
    FLUSH_HOUSE = 10
    STRAIGHT_FLUSH = 11
    FLUSH_FIVE = 12


@dataclass
class HandResult:
    """Comparable result of evaluating a poker hand."""

    rank: HandRank
    description: str
    cards: List[Card]
    kickers: List[int]

    def _cmp_key(self) -> Tuple[int, Tuple[int, ...]]:
        return self.rank.value, tuple(self.kickers)

    def __lt__(self, other: "HandResult") -> bool:
        return self._cmp_key() < other._cmp_key()

    def __le__(self, other: "HandResult") -> bool:
        return self._cmp_key() <= other._cmp_key()

    def __gt__(self, other: "HandResult") -> bool:
        return self._cmp_key() > other._cmp_key()

    def __ge__(self, other: "HandResult") -> bool:
        return self._cmp_key() >= other._cmp_key()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandResult):
            return NotImplemented
        return self._cmp_key() == other._cmp_key()


class InfiniteHandEvaluator:
    """Evaluate hands for the infinite replacement-dealing variant."""

    @staticmethod
    def evaluate(cards: List[Card]) -> HandResult:
        if len(cards) < 5:
            raise ValueError(f"Need at least 5 cards, got {len(cards)}")

        best_hand = None
        for five_cards in combinations(cards, 5):
            hand = InfiniteHandEvaluator._evaluate_five(list(five_cards))
            if best_hand is None or hand > best_hand:
                best_hand = hand

        return best_hand

    @staticmethod
    def _evaluate_five(cards: List[Card]) -> HandResult:
        rank_counts = Counter(card.rank for card in cards)
        suit_counts = Counter(card.suit for card in cards)
        rank_values = sorted((card.rank_value for card in cards), reverse=True)
        is_flush = max(suit_counts.values()) == 5
        is_straight = InfiniteHandEvaluator._is_straight(rank_values)
        max_rank_count = max(rank_counts.values())
        rank_count_values = sorted(rank_counts.values(), reverse=True)

        card_counts = Counter((card.rank, card.suit) for card in cards)
        if max(card_counts.values()) == 5:
            card = cards[0]
            return HandResult(
                HandRank.FLUSH_FIVE,
                f"Flush Five, {RANK_NAMES[card.rank]}",
                cards,
                [RANK_VALUES[card.rank]],
            )

        if is_flush and is_straight:
            high = max(rank_values) if not InfiniteHandEvaluator._is_wheel(rank_values) else 3
            return HandResult(
                HandRank.STRAIGHT_FLUSH,
                f"Straight Flush, {RANKS[high]} high",
                cards,
                [high],
            )

        if is_flush and rank_count_values == [3, 2]:
            trips_rank = next(rank for rank, count in rank_counts.items() if count == 3)
            pair_rank = next(rank for rank, count in rank_counts.items() if count == 2)
            return HandResult(
                HandRank.FLUSH_HOUSE,
                f"Flush House, {RANK_NAMES[trips_rank]} full of {RANK_NAMES[pair_rank]}",
                cards,
                [RANK_VALUES[trips_rank], RANK_VALUES[pair_rank]],
            )

        if max_rank_count == 5:
            rank = next(rank for rank, count in rank_counts.items() if count == 5)
            return HandResult(
                HandRank.FIVE_OF_A_KIND,
                f"Five of a Kind, {RANK_NAMES[rank]}",
                cards,
                [RANK_VALUES[rank]],
            )

        if max_rank_count == 4:
            quad_rank = next(rank for rank, count in rank_counts.items() if count == 4)
            kicker = next(rank for rank, count in rank_counts.items() if count != 4)
            return HandResult(
                HandRank.FOUR_OF_A_KIND,
                f"Four of a Kind, {RANK_NAMES[quad_rank]}",
                cards,
                [RANK_VALUES[quad_rank], RANK_VALUES[kicker]],
            )

        if is_straight:
            high = max(rank_values) if not InfiniteHandEvaluator._is_wheel(rank_values) else 3
            return HandResult(
                HandRank.STRAIGHT,
                f"Straight, {RANKS[high]} high",
                cards,
                [high],
            )

        if is_flush:
            return HandResult(
                HandRank.FLUSH,
                f"Flush, {RANKS[rank_values[0]]} high",
                cards,
                rank_values,
            )

        if rank_count_values == [3, 2]:
            trips_rank = next(rank for rank, count in rank_counts.items() if count == 3)
            pair_rank = next(rank for rank, count in rank_counts.items() if count == 2)
            return HandResult(
                HandRank.FULL_HOUSE,
                f"Full House, {RANK_NAMES[trips_rank]} full of {RANK_NAMES[pair_rank]}",
                cards,
                [RANK_VALUES[trips_rank], RANK_VALUES[pair_rank]],
            )

        if max_rank_count == 3:
            trips_rank = next(rank for rank, count in rank_counts.items() if count == 3)
            kickers = sorted(
                (RANK_VALUES[rank] for rank, count in rank_counts.items() if count != 3),
                reverse=True,
            )
            return HandResult(
                HandRank.THREE_OF_A_KIND,
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
            return HandResult(
                HandRank.TWO_PAIR,
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
            return HandResult(
                HandRank.PAIR,
                f"Pair of {RANK_NAMES[pair_rank]}",
                cards,
                [RANK_VALUES[pair_rank]] + kickers,
            )

        return HandResult(
            HandRank.HIGH_CARD,
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


class InfiniteHoldemGame:
    """Heads-up infinite hold'em engine with real betting-round control."""

    def __init__(
        self,
        num_players: int = 2,
        starting_stack: int = 10000,
        small_blind: int = 50,
        big_blind: int = 100,
        seed: Optional[int] = None,
    ):
        if num_players != 2:
            raise ValueError("Phase 1 supports heads-up play only")

        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.deck = InfiniteDeck(seed)
        self.evaluator = InfiniteHandEvaluator()
        self.new_hand()

    def new_hand(self) -> None:
        """Reset to a fresh independent heads-up hand."""

        self.stacks = [self.starting_stack] * self.num_players
        self.hole_cards: List[List[Card]] = [[] for _ in range(self.num_players)]
        self.board: List[Card] = []
        self.pot = 0
        self.current_bets = [0] * self.num_players
        self.total_invested = [0] * self.num_players
        self.folded = [False] * self.num_players
        self.street = "preflop"
        self.actor_index: Optional[int] = None
        self.last_raise_size = self.big_blind
        self.hand_over = False
        self._pending_actors: List[int] = []
        self._settled = False
        self._settlement: Optional[Tuple[List[int], List[float]]] = None

        self._post_blind(0, self.small_blind)
        self._post_blind(1, self.big_blind)
        self._open_betting_round()

    reset = new_hand

    def _post_blind(self, player_idx: int, amount: int) -> None:
        actual = min(amount, self.stacks[player_idx])
        self.stacks[player_idx] -= actual
        self.pot += actual
        self.current_bets[player_idx] += actual
        self.total_invested[player_idx] += actual

    def _active_players(self) -> List[int]:
        return [player for player in range(self.num_players) if not self.folded[player]]

    def _street_order(self) -> List[int]:
        order = [0, 1] if self.street == "preflop" else [1, 0]
        return [player for player in order if not self.folded[player]]

    def _open_betting_round(self) -> None:
        order = self._street_order()
        self._pending_actors = order[:]
        self.actor_index = order[0] if order else None

    def _require_actor(self, player_idx: int) -> None:
        if self.hand_over:
            raise ValueError("The hand is already over")
        if player_idx != self.actor_index:
            raise ValueError(f"Player {player_idx} is not the current actor")

    def _consume_actor(self, player_idx: int) -> None:
        self._require_actor(player_idx)
        if self._pending_actors and self._pending_actors[0] == player_idx:
            self._pending_actors.pop(0)
        elif player_idx in self._pending_actors:
            self._pending_actors.remove(player_idx)

        if len(self._active_players()) <= 1:
            self.actor_index = None
            self._pending_actors = []
            self.hand_over = True
            return

        if self._pending_actors:
            self.actor_index = self._pending_actors[0]
            return

        self.actor_index = None
        if self.street == "river":
            self.street = "showdown"
            self.hand_over = True

    def deal_hole_cards(self) -> None:
        for player_idx in range(self.num_players):
            self.hole_cards[player_idx] = self.deck.deal(2)

    def deal_flop(self) -> None:
        if self.street != "preflop":
            raise ValueError("Flop can only be dealt after preflop")
        self.board.extend(self.deck.deal(3))
        self.street = "flop"
        self.current_bets = [0] * self.num_players
        self.last_raise_size = self.big_blind
        self._open_betting_round()

    def deal_turn(self) -> None:
        if self.street != "flop":
            raise ValueError("Turn can only be dealt after the flop")
        self.board.extend(self.deck.deal(1))
        self.street = "turn"
        self.current_bets = [0] * self.num_players
        self.last_raise_size = self.big_blind
        self._open_betting_round()

    def deal_river(self) -> None:
        if self.street != "turn":
            raise ValueError("River can only be dealt after the turn")
        self.board.extend(self.deck.deal(1))
        self.street = "river"
        self.current_bets = [0] * self.num_players
        self.last_raise_size = self.big_blind
        self._open_betting_round()

    def advance_street(self) -> None:
        if self.hand_over:
            raise ValueError("The hand is already over")
        if not self.is_betting_round_complete():
            raise ValueError("Cannot advance streets before the betting round is complete")

        if self.street == "preflop":
            self.deal_flop()
        elif self.street == "flop":
            self.deal_turn()
        elif self.street == "turn":
            self.deal_river()
        elif self.street == "river":
            self.street = "showdown"
            self.hand_over = True
        else:
            raise ValueError(f"Cannot advance from street {self.street}")

    def get_to_call(self, player_idx: int) -> int:
        return max(self.current_bets) - self.current_bets[player_idx]

    def get_min_raise_to(self, player_idx: Optional[int] = None) -> Optional[int]:
        if player_idx is None:
            player_idx = self.actor_index
        if player_idx is None or self.hand_over or self.folded[player_idx]:
            return None

        max_raise_to = self.get_max_raise_to(player_idx)
        if max_raise_to is None:
            return None

        current_high_bet = max(self.current_bets)
        if current_high_bet == 0:
            min_raise_to = self.current_bets[player_idx] + self.big_blind
        else:
            min_raise_to = current_high_bet + self.last_raise_size

        if max_raise_to < min_raise_to:
            return None

        return min_raise_to

    def get_max_raise_to(self, player_idx: Optional[int] = None) -> Optional[int]:
        if player_idx is None:
            player_idx = self.actor_index
        if player_idx is None or self.hand_over or self.folded[player_idx]:
            return None
        return self.current_bets[player_idx] + self.stacks[player_idx]

    def get_legal_actions(self, player_idx: Optional[int] = None) -> List[str]:
        if player_idx is None:
            player_idx = self.actor_index
        if (
            player_idx is None
            or self.hand_over
            or self.folded[player_idx]
            or player_idx != self.actor_index
        ):
            return []

        legal_actions: List[str] = []
        to_call = self.get_to_call(player_idx)
        if to_call == 0:
            legal_actions.append("check")
        else:
            legal_actions.extend(["fold", "call"])

        if self.get_min_raise_to(player_idx) is not None:
            legal_actions.append("raise")

        return legal_actions

    def fold(self, player_idx: int) -> None:
        self._require_actor(player_idx)
        self.folded[player_idx] = True
        self._pending_actors = [player for player in self._pending_actors if player != player_idx]
        self.actor_index = None
        self.hand_over = True

    def check(self, player_idx: int) -> None:
        if self.get_to_call(player_idx) != 0:
            raise ValueError("Cannot check while facing a bet")
        self._consume_actor(player_idx)

    def call(self, player_idx: int) -> int:
        to_call = self.get_to_call(player_idx)
        if to_call == 0:
            self.check(player_idx)
            return 0

        self._require_actor(player_idx)
        actual = min(to_call, self.stacks[player_idx])
        self.stacks[player_idx] -= actual
        self.pot += actual
        self.current_bets[player_idx] += actual
        self.total_invested[player_idx] += actual
        self._consume_actor(player_idx)
        return actual

    def bet_or_raise(self, player_idx: int, total_amount: int) -> int:
        self._require_actor(player_idx)
        min_raise_to = self.get_min_raise_to(player_idx)
        max_raise_to = self.get_max_raise_to(player_idx)
        if min_raise_to is None or max_raise_to is None:
            raise ValueError("Raising is not legal in the current state")
        if total_amount < min_raise_to or total_amount > max_raise_to:
            raise ValueError(
                f"Raise must be between {min_raise_to} and {max_raise_to}, got {total_amount}"
            )

        current_high_bet = max(self.current_bets)
        additional = total_amount - self.current_bets[player_idx]
        self.stacks[player_idx] -= additional
        self.pot += additional
        self.current_bets[player_idx] = total_amount
        self.total_invested[player_idx] += additional
        self.last_raise_size = total_amount - current_high_bet

        self._pending_actors = [
            player
            for player in self._street_order()
            if player != player_idx
        ]
        self.actor_index = self._pending_actors[0] if self._pending_actors else None
        return additional

    def apply_action(
        self,
        player_idx: int,
        action: str,
        total_amount: Optional[int] = None,
    ) -> Optional[int]:
        if action == "fold":
            self.fold(player_idx)
            return None
        if action == "check":
            self.check(player_idx)
            return None
        if action == "call":
            return self.call(player_idx)
        if action == "raise":
            if total_amount is None:
                raise ValueError("Raise actions require a total-to amount")
            return self.bet_or_raise(player_idx, total_amount)
        raise ValueError(f"Unknown action: {action}")

    def is_betting_round_complete(self) -> bool:
        return not self.hand_over and self.actor_index is None and not self._pending_actors

    def evaluate_hand(self, player_idx: int) -> HandResult:
        return self.evaluator.evaluate(self.hole_cards[player_idx] + self.board)

    def determine_winner(self) -> Tuple[List[int], List[Optional[HandResult]]]:
        active_players = self._active_players()
        results: List[Optional[HandResult]] = [None] * self.num_players

        if len(active_players) == 1:
            return active_players, results

        ranked_hands = []
        for player_idx in active_players:
            result = self.evaluate_hand(player_idx)
            results[player_idx] = result
            ranked_hands.append((player_idx, result))

        ranked_hands.sort(key=lambda item: item[1], reverse=True)
        best_hand = ranked_hands[0][1]
        winners = [player_idx for player_idx, result in ranked_hands if result == best_hand]
        return winners, results

    def settle_pot(self) -> Tuple[List[int], List[float]]:
        if self._settled and self._settlement is not None:
            return self._settlement

        if not self.hand_over:
            raise ValueError("Cannot settle the pot before the hand is over")

        winners, _ = self.determine_winner()
        share = self.pot // len(winners)
        remainder = self.pot % len(winners)

        for index, winner in enumerate(winners):
            award = share + (1 if index < remainder else 0)
            self.stacks[winner] += award

        payoffs = [stack - self.starting_stack for stack in self.stacks]
        self.pot = 0
        self._settlement = (winners, payoffs)
        self._settled = True
        return self._settlement

    def is_hand_over(self) -> bool:
        return self.hand_over

