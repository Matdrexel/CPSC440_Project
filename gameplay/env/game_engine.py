"""
game_engine.py — Heads-up no-limit Texas Hold'em game engine.

Handles the full game loop: blinds, dealing, betting rounds, showdown,
and reward assignment. Designed to be wrapped by poker_env.py (Gym).

Streets:   0=preflop, 1=flop, 2=turn, 3=river
Players:   0=BTN/SB (acts first preflop, second postflop)
           1=BB     (acts second preflop, first postflop)
"""

import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from env.deck import deal_card
from env.hand_evaluator import check_hand


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class Street(IntEnum):
    PREFLOP = 0
    FLOP    = 1
    TURN    = 2
    RIVER   = 3

class Action(IntEnum):
    FOLD  = 0
    CALL  = 1
    RAISE = 2   # raise actions start here; index into raise_sizes via (action - 2)

BOARD_SIZE = [0, 3, 1, 1]   # cards dealt per street transition


# ---------------------------------------------------------------------------
# GameState dataclass — full snapshot of a hand
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    # Config
    starting_stack: int
    small_blind: int
    big_blind: int
    raise_sizes: list[float]        # e.g. [0.5, 1.0, 2.0] = fractions of pot

    # Cards
    hole_cards: list[list[int]] = field(default_factory=lambda: [[], []])
    board: list[int] = field(default_factory=list)

    # Chips
    stacks: list[int] = field(default_factory=list)
    pot: int = 0
    bets: list[int] = field(default_factory=lambda: [0, 0])             # chips put in THIS street
    total_contributed: list[int] = field(default_factory=lambda: [0, 0]) # chips put in across ALL streets

    # State
    street: int = Street.PREFLOP
    acting_player: int = 0          # whose turn it is
    street_acted: list[bool] = field(default_factory=lambda: [False, False])
    hand_over: bool = False
    winner: int = -1                # -1 = not yet decided / chop
    rewards: list[float] = field(default_factory=lambda: [0.0, 0.0])
    # rewards[i] = net chips won or lost by player i over the entire hand.
    # Defined as: (chips returned at end) - (chips contributed to pot total).
    # Always zero-sum: rewards[0] + rewards[1] == 0.
    # Examples:
    #   SB folds preflop            → rewards = [-5,   +5]
    #   BB folds to a raise (put in 30 total) → rewards = [+30, -30]
    #   Showdown: each put in 100, winner takes 200-chip pot → rewards = [+100, -100]
    # Only populated once hand_over is True.

    @property
    def current_call_amount(self) -> int:
        """How much the acting player needs to add to match the largest bet."""
        return max(self.bets) - self.bets[self.acting_player]

    @property
    def total_actions(self) -> int:
        return 2 + len(self.raise_sizes)


# ---------------------------------------------------------------------------
# GameEngine
# ---------------------------------------------------------------------------

class GameEngine:
    """
    Manages a single heads-up no-limit Texas Hold'em hand.

    Usage:
        engine = GameEngine(starting_stack=1000, small_blind=5,
                            big_blind=10, raise_sizes=[0.5, 1.0, 2.0])
        state = engine.reset()
        while not state.hand_over:
            action = agent.act(state)
            state = engine.step(action)
    """

    def __init__(self,
                 starting_stack: int = 1000,
                 small_blind: int = 5,
                 big_blind: int = 10,
                 raise_sizes: list[float] = None):
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.raise_sizes = raise_sizes or [0.5, 1.0, 2.0]
        self.state: GameState = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> GameState:
        """Start a new hand. Returns the initial GameState."""
        self.state = GameState(
            starting_stack=self.starting_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            raise_sizes=self.raise_sizes,
            stacks=[self.starting_stack, self.starting_stack],
        )
        self._deal_hole_cards()
        self._post_blinds()
        return self.state

    def step(self, action: int) -> GameState:
        """
        Apply `action` for the current acting player.
        action: 0=fold, 1=call, 2+=raise (index into raise_sizes via action-2)
        Returns updated GameState.
        """
        if self.state.hand_over:
            raise RuntimeError("Hand is over — call reset() to start a new hand.")

        if action == Action.FOLD:
            self._handle_fold()
        elif action == Action.CALL:
            self._handle_call()
        else:
            raise_idx = action - int(Action.RAISE)
            if raise_idx < 0 or raise_idx >= len(self.raise_sizes):
                raise ValueError(f"Invalid raise index {raise_idx}. "
                                 f"Valid actions: 0..{1 + len(self.raise_sizes)}")
            self._handle_raise(self.raise_sizes[raise_idx])

        if not self.state.hand_over:
            self._maybe_advance_street()

        return self.state

    def legal_actions(self) -> list[int]:
        """Return list of legal action indices for the current acting player."""
        actions = [Action.FOLD, Action.CALL]
        player = self.state.acting_player
        stack = self.state.stacks[player]
        call_amount = self.state.current_call_amount

        # Can only raise if there are chips left after calling
        remaining_after_call = stack - call_amount
        if remaining_after_call > 0:
            for i, frac in enumerate(self.raise_sizes):
                raise_amount = self._raise_amount(frac)
                if raise_amount >= self.big_blind and raise_amount <= stack:
                    actions.append(int(Action.RAISE) + i)

        return actions
    
    def legal_actions_mask(self) -> "np.ndarray":
        """
        Return a boolean numpy array of shape (n_actions,) where True means
        the action is legal. n_actions = 2 + len(raise_sizes).
        Suitable for directly masking logits before softmax in the PPO agent.
        """
        n = 2 + len(self.raise_sizes)
        mask = np.zeros(n, dtype=bool)
        for a in self.legal_actions():
            mask[a] = True
        return mask

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_fold(self):
        s = self.state
        loser = s.acting_player
        winner = 1 - loser
        s.winner = winner
        s.stacks[winner] += s.pot
        s.pot = 0
        # Net chips gained/lost relative to total contribution across the whole hand
        s.rewards[winner] = float(s.total_contributed[loser])
        s.rewards[loser]  = float(-s.total_contributed[loser])
        s.hand_over = True

    def _handle_call(self):
        s = self.state
        player = s.acting_player
        amount = min(s.current_call_amount, s.stacks[player])
        s.stacks[player]           -= amount
        s.bets[player]             += amount
        s.total_contributed[player] += amount
        s.pot                      += amount
        s.street_acted[player]      = True

    def _handle_raise(self, pot_fraction: float):
        s = self.state
        player = s.acting_player

        # First, call the current bet, then add the raise on top
        call_amount  = min(s.current_call_amount, s.stacks[player])
        raise_amount = min(self._raise_amount(pot_fraction), s.stacks[player] - call_amount)
        total = call_amount + raise_amount

        s.stacks[player]           -= total
        s.bets[player]             += total
        s.total_contributed[player] += total
        s.pot                      += total

        # Reset acted flags so opponent must respond
        s.street_acted[player]     = True
        s.street_acted[1 - player] = False

    def _raise_amount(self, pot_fraction: float) -> int:
        """Compute raise size as a fraction of the current pot."""
        return max(self.big_blind, int(self.state.pot * pot_fraction))

    # ------------------------------------------------------------------
    # Street management
    # ------------------------------------------------------------------

    def _maybe_advance_street(self):
        s = self.state
        both_acted = all(s.street_acted)
        bets_equal = s.bets[0] == s.bets[1]

        if not (both_acted and bets_equal):
            # Action continues — pass to next player
            s.acting_player = 1 - s.acting_player
            return

        if s.street == Street.RIVER:
            self._showdown()
        else:
            self._advance_to_next_street()

    def _advance_to_next_street(self):
        s = self.state
        s.street += 1
        # Deal community cards
        for _ in range(BOARD_SIZE[s.street]):
            s.board.append(deal_card())
        # Reset per-street state
        s.bets = [0, 0]
        s.street_acted = [False, False]
        # Postflop: BB (player 1) acts first
        s.acting_player = 1

    def _showdown(self):
        s = self.state
        all_cards_0 = s.hole_cards[0] + s.board
        all_cards_1 = s.hole_cards[1] + s.board

        hand_0 = check_hand(all_cards_0)
        hand_1 = check_hand(all_cards_1)

        if hand_0 > hand_1:
            s.winner = 0
        elif hand_1 > hand_0:
            s.winner = 1
        else:
            s.winner = -1   # chop

        # Distribute pot and assign rewards based on total contribution across all streets
        if s.winner == -1:
            half = s.pot // 2
            s.stacks[0] += half
            s.stacks[1] += s.pot - half   # odd chip goes to player 1
            s.rewards[0] = float(half - s.total_contributed[0])
            s.rewards[1] = float((s.pot - half) - s.total_contributed[1])
        else:
            s.stacks[s.winner] += s.pot
            s.rewards[s.winner]     = float(s.pot - s.total_contributed[s.winner])
            s.rewards[1 - s.winner] = float(-s.total_contributed[1 - s.winner])

        s.pot = 0
        s.hand_over = True

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _deal_hole_cards(self):
        for p in range(2):
            self.state.hole_cards[p] = [deal_card(), deal_card()]

    def _post_blinds(self):
        s = self.state
        # Player 0 = BTN/SB posts small blind
        sb = min(self.small_blind, s.stacks[0])
        s.stacks[0]            -= sb
        s.bets[0]               = sb
        s.total_contributed[0]  = sb
        s.pot                  += sb

        # Player 1 = BB posts big blind
        bb = min(self.big_blind, s.stacks[1])
        s.stacks[1]            -= bb
        s.bets[1]               = bb
        s.total_contributed[1]  = bb
        s.pot                  += bb

        # Preflop: SB (player 0) acts first
        s.acting_player = 0
        # BB has already "acted" by posting — but SB can still raise,
        # so we mark BB as not having had a voluntary action yet.
        s.street_acted = [False, False]